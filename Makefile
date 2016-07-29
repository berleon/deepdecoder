
output_dir=$(shell realpath out)


get = $(shell cat $(2) | shyaml get-value $(1))

test_get:
	echo '{"hello": "world"}' > /tmp/test_get
	test "$(call get,hello,/tmp/test_get)" = "world"

.PHONY: create_output_dir
create_output_dir:
	mkdir -p $(output_dir)

real_images_dir = $(shell realpath $(call get,image_directory,real_images_setting.yaml))
real_images: real_images_setting.yaml
	mkdir -p $(output_dir)
	echo "Real image directory is: $(real_images_dir)"
	find `realpath $(real_images_dir)` -name "*.jpeg" > real_images


localizer_preprocessed_images_dir = $(output_dir)/localizer_preprocessed
localizer_preprocessed_images_list= $(localizer_preprocessed_images_dir)/images.txt
localizer_preprocessed_images: real_images localizer_preprocessed_images_setting.yaml
	bb_preprocess \
            --output-dir $(localizer_preprocessed_images_dir) \
            --output-pathfile $(localizer_preprocessed_images_list) \
            --pathfile real_images \
            --border $(call get,border,localizer_preprocessed_images_setting.yaml) \
            --use-hist-eq $(call get,use_clahe,localizer_preprocessed_images_setting.yaml) \
            --format jpeg \
            --compression 90
	echo  '{"output_dir": "$(localizer_preprocessed_images_dir)",'\
		'"output_pathfile": "$(localizer_preprocessed_images_list)"}' > localizer_preprocessed_images

localizer_weights_dir =$(output_dir)/localizer_weights_repo
localizer_weights: localizer_settings.yaml
	rm -rf $(localizer_weights_dir)
	git clone  --depth 1 $(call get,repo,localizer_settings.yaml) $(localizer_weights_dir)
	echo  '{"weights": "$(localizer_weights_dir)/$(call get,weights,localizer_settings.yaml)"}' > localizer_weights

FIND_TAGS_JSON=$(output_dir)/real_tags_positions.json
find_tags: find_tags_setting.yaml localizer_weights localizer_preprocessed_images
	bb_find_tags --out $(FIND_TAGS_JSON) \
			--weight  $(call get,weights,localizer_weights) \
			--threshold $(call get,saliency_threshold,find_tags_setting.yaml) \
			$(call get,output_pathfile,localizer_preprocessed_images)
	echo  '{"tags_position": "$(FIND_TAGS_JSON)"}' > find_tags


build_real_dataset_hdf5 =$(output_dir)/real_tags.hdf5
real_dataset=
build_real_dataset: find_tags real_images build_real_dataset_setting.yaml
	rm -rf $(build_real_dataset_hdf5)
	bb_build_tag_dataset \
             --out "$(build_real_dataset_hdf5)" \
             --roi-size $(call get,roi_size,build_real_dataset_setting.yaml) \
             --offset $(call get,offset,build_real_dataset_setting.yaml) \
             --threshold $(call get,threshold,build_real_dataset_setting.yaml) \
             --image-pathfile real_images \
             $(call get,tags_position,find_tags)
	echo '{"path": "$(build_real_dataset_hdf5)"}' > build_real_dataset
generate_3d_tags_distribution.json:
	bb_default_3d_tags_distribution generate_3d_tags_distribution.json


generated_3d_tags_dir=$(output_dir)/generated_3d_tags/
generate_3d_tag_samples=$(call get,nb_samples,generate_3d_tags_settings.yaml)
generate_3d_tags: generate_3d_tags_distribution.json generate_3d_tags_settings.yaml
	bb_generate_3d_tags \
		--force \
		--nb-samples $(generate_3d_tag_samples)  \
		--dist generate_3d_tags_distribution.json \
		$(generated_3d_tags_dir)/generated_3d_tags.hdf5
	echo '{"path": "$(generated_3d_tags_dir)/generated_3d_tags.hdf5"}' > generate_3d_tags


network_3d_tag_dir = $(output_dir)/network_3d_tag
n3d_units = $(call get,units,network_3d_tag_settings.yaml)
n3d_depth = $(call get,depth,network_3d_tag_settings.yaml)
n3d_epoch = $(call get,epoch,network_3d_tag_settings.yaml)
n3d_nb_dense = $(call get,nb_dense,network_3d_tag_settings.yaml)
n3d_weights =$(network_3d_tag_dir)/network_tags3d_n$(n3d_units)_d$(n3d_depth)_e$(n3d_epoch).hdf5
tag3d_network_weights=$(call,get,weights_path,train_tag3d_network)
train_tag3d_network: generate_3d_tags network_3d_tag_settings.yaml
	bb_train_tag3d_network \
		--force \
		--3d-tags $(call get,path,generate_3d_tags) \
		--units $(n3d_units) \
		--depth $(n3d_depth)\
		--epoch $(n3d_epoch) \
		--nb-dense $(n3d_nb_dense) \
		$(network_3d_tag_dir)
	test -e $(n3d_weights)
	echo '{"weights_path": "$(n3d_weights)"}' > train_tag3d_network

generator_units = $(call get,gen-units,render_gan_settings.yaml)
discriminator_units = $(call get,dis-units,render_gan_settings.yaml)
render_gan_dir = $(output_dir)/render_gan_morning
train_render_gan: train_tag3d_network build_real_dataset
	bb_train_rendergan \
		--real $(call get,path,build_real_dataset) \
		--nntag3d $(call get,weights_path,train_tag3d_network) \
		--output-dir $(render_gan_dir) \
		--force  \
		--epoch $(call get,epoch,render_gan_settings.yaml) \
		--dis-units $(discriminator_units) \
		--gen-units $(generator_units)

sample_artificial_trainset: train_render_gan
	bb_sample_artificial_trainset

train_decoder:
	bb_train_decoder

evaluate_decoder:
	bb_evaluate_decoder
