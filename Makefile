
output_dir=$(shell realpath out)

CONFIG_FILE="train.json"

real_images_dir="real_images"

config = $(call get,$(1),$(CONFIG_FILE))
get = $(shell jq --raw-output .$(1) $(2))


.PHONY: create_output_dir
create_output_dir:
	mkdir -p $(output_dir)

real_images_dir = $(shell realpath $(call get,image_directory,real_images_setting.json))
real_images: real_images_setting.json
	mkdir -p $(output_dir)
	echo "Real image directory is: $(real_images_dir)"
	find `realpath $(real_images_dir)` -name "*.jpeg" > real_images


localizer_preprocessed_images_dir = $(output_dir)/localizer_preprocessed
localizer_preprocessed_images_list= $(localizer_preprocessed_images_dir)/images.txt
localizer_preprocessed_images: real_images localizer_preprocessed_images_setting.json
	bb_preprocess \
            --output-dir $(localizer_preprocessed_images_dir) \
            --output-pathfile $(localizer_preprocessed_images_list) \
            --pathfile real_images \
            --border $(call get,border,localizer_preprocessed_images_setting.json) \
            --use-hist-eq $(call get,use_clahe,localizer_preprocessed_images_setting.json) \
            --format jpeg \
            --compression 90
	echo  '{"output_dir": "$(localizer_preprocessed_images_dir)",'\
		'"output_pathfile": "$(localizer_preprocessed_images_list)"}' > localizer_preprocessed_images

localizer_weights_dir =$(output_dir)/localizer_weights_repo
localizer_weights: localizer_settings.json
	rm -rf $(localizer_weights_dir)
	git clone  --depth 1 $(call get,repo,localizer_settings.json) $(localizer_weights_dir)
	echo  '{"weights": "$(localizer_weights_dir)/$(call get,weights,localizer_settings.json)"}' > localizer_weights

FIND_TAGS_JSON=$(output_dir)/real_tags_positions.json
find_tags: find_tags_setting.json localizer_weights localizer_preprocessed_images
	bb_find_tags --out $(FIND_TAGS_JSON) \
			--weight  $(call get,weights,localizer_weights) \
			--threshold $(call get,saliency_threshold,find_tags_setting.json) \
			$(call get,output_pathfile,localizer_preprocessed_images)
	echo  '{"tags_position": "$(FIND_TAGS_JSON)"}' > find_tags


build_real_dataset_hdf5 ="$(output_dir)/real_tags.hdf5"
build_real_dataset: find_tags real_images build_real_dataset_setting.json
	rm -rf $(build_real_dataset_hdf5)
	bb_build_tag_dataset \
             --out $(build_real_dataset_hdf5) \
             --roi-size $(call get,roi_size,build_real_dataset_setting.json) \
             --offset $(call get,offset,build_real_dataset_setting.json) \
             --threshold $(call get,threshold,build_real_dataset_setting.json) \
             --image-pathfile real_images \
             $(call get,tags_position,find_tags) \

generate_3d_tags_distribution.json:
	bb_default_3d_tags_distribution generate_3d_tags_distribution.json


generated_3d_tags_dir=$(output_dir)/generated_3d_tags/
generate_3d_tag_samples=$(call get,nb_samples,generate_3d_tags_settings.json)
generate_3d_tags: generate_3d_tags_distribution.json generate_3d_tags_settings.json
	bb_generate_3d_tags \
		--force \
		--nb-samples $(generate_3d_tag_samples)  \
		--dist generate_3d_tags_distribution.json \
		$(generated_3d_tags_dir)/generated_3d_tags.hdf5
	echo '{"path": "$(generated_3d_tags_dir)/generated_3d_tags.hdf5"}' > generate_3d_tags


network_3d_tag_dir = $(output_dir)/network_3d_tag
n3d_units = $(call get,units,network_3d_tag_settings.json)
n3d_depth = $(call get,depth,network_3d_tag_settings.json)
n3d_epoch = $(call get,epoch,network_3d_tag_settings.json)
n3d_weights =$(network_3d_tag_dir)/network_3d_tags_n$(n3d_units)_d$(n3d_depth)_e$(n3d_epoch).hdf5

train_network_3d_tag: generate_3d_tags network_3d_tag_settings.json
	bb_train_3d_tags_network \
		--force \
		--3d-tags $(call get,path,generate_3d_tags) \
		--units $(n3d_units) \
		--depth $(n3d_depth)\
		--epoch $(n3d_epoch) \
		$(network_3d_tag_dir)
	test -e $(n3d_weights)
	echo '{"weights_path": "$(n3d_weights)"}' > train_network_3d_tag

train_render_gan: train_structure_model, build_real_dataset
	bb_train_render_gan


sample_artificial_trainset: train_render_gan
	bb_sample_artificial_trainset

train_decoder:
	bb_train_decoder

evaluate_decoder:
	bb_evaluate_decoder
