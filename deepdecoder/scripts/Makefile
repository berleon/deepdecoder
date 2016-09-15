
data=$(shell realpath data)
models=$(shell realpath models)
ensure_out_dirs_exists = mkdir -p $(data) $(models)

setting = $(shell cat $(setting_dir)/$(2) | shyaml get-value $(1))
get = $(shell cat $(2) | shyaml get-value $(1))

test_get:
	echo '{"hello": "world"}' > /tmp/test_get
	test "$(call get,hello,/tmp/test_get)" = "world"


REAL_TAGS_DATASET=$(data)/real_tags.hdf5
real_dataset: $(setting_dir)/real_images.txt
	$(ensure_out_dirs_exists)
	bb_build_real_tag_dataset --out $(REAL_TAGS_DATASET) \
		--force \
		$(setting_dir)/real_images.txt
	echo  '{"path": "$(REAL_TAGS_DATASET)"}' > real_dataset


generate_3d_tags_distribution.json:
	bb_default_3d_tags_distribution generate_3d_tags_distribution.json


generated_3d_tags_dir=$(data)/generated_3d_tags/
generate_3d_tag_samples=$(call setting,nb_samples,tag3d_data_set.yaml)
tag3d_data_set: generate_3d_tags_distribution.json $(setting_dir)/tag3d_data_set.yaml
	$(ensure_out_dirs_exists)
	bb_generate_3d_tags \
		--force \
		--nb-samples $(generate_3d_tag_samples)  \
		--dist generate_3d_tags_distribution.json \
		$(generated_3d_tags_dir)/generated_3d_tags.hdf5
	echo '{"path": "$(generated_3d_tags_dir)/generated_3d_tags.hdf5"}' > tag3d_data_set


network_3d_tag_dir = $(models)/tag3d_network
n3d_units = $(call setting,units,tag3d_network.yaml)
n3d_depth = $(call setting,depth,tag3d_network.yaml)
n3d_epoch = $(call setting,epoch,tag3d_network.yaml)
n3d_nb_dense = $(call setting,nb_dense,tag3d_network.yaml)
n3d_weights =$(network_3d_tag_dir)/network_tags3d_n$(n3d_units)_d$(n3d_depth)_e$(n3d_epoch).hdf5
tag3d_network: tag3d_data_set $(setting_dir)/tag3d_network.yaml
	$(ensure_out_dirs_exists)
	bb_train_tag3d_network \
		--force \
		--3d-tags $(call get,path,tag3d_data_set) \
		--units $(n3d_units) \
		--depth $(n3d_depth)\
		--epoch $(n3d_epoch) \
		--nb-dense $(n3d_nb_dense) \
		$(network_3d_tag_dir)
	test -e $(n3d_weights)
	echo '{"weights_path": "$(n3d_weights)"}' > tag3d_network

generator_units = $(call setting,gen-units,rendergan.yaml)
discriminator_units = $(call setting,dis-units,rendergan.yaml)
rendergan_dir = $(models)/rendergan/
rendergan: tag3d_network real_dataset $(setting_dir)/rendergan.yaml
	$(ensure_out_dirs_exists)
	bb_train_rendergan \
		--real $(call get,path,real_dataset) \
		--nntag3d $(call get,weights_path,tag3d_network) \
		--output-dir $(rendergan_dir) \
		--force  \
		--epoch $(call setting,epoch,rendergan.yaml) \
		--dis-units $(discriminator_units) \
		--gen-units $(generator_units)
	echo '{"path": "${rendergan_dir}", "epoch": "300"}' > rendergan

render_gan_models_dir = $(call get,path,rendergan)/models/$(call setting,epoch,artificial_data_set.yaml)/
artificial_data_set: rendergan $(setting_dir)/artificial_data_set.yaml
	$(ensure_out_dirs_exists)
	$(eval artificial_data_set_date := $(shell date -Iseconds))
	bb_sample_from_rendergan \
		--generator ${render_gan_models_dir}/sample_generator_given_z.hdf5 \
		--discriminator ${render_gan_models_dir}/discriminator.hdf5 \
		--nb-samples $(call setting,nb_train_samples,artificial_data_set.yaml) \
		--selected-outputs $(call setting,selected_outputs,artificial_data_set.yaml) \
		${data}/artificial_train_set_$(artificial_data_set_date).hdf5
	bb_sample_from_rendergan \
		--generator ${render_gan_models_dir}/sample_generator_given_z.hdf5 \
		--discriminator ${render_gan_models_dir}/discriminator.hdf5 \
		--nb-samples $(call setting,nb_test_samples,artificial_data_set.yaml) \
		--selected-outputs $(call setting,selected_outputs,artificial_data_set.yaml) \
		${data}/artificial_test_set_$(artificial_data_set_date).hdf5
	echo '{"train_set": "${data}/artificial_train_set.hdf5",\n' \
		 '"test_set": "${data}/artificial_test_set.hdf5"}' \
		 > artificial_data_set

decoder_model: artificial_data_set $(setting_dir)/decoder.yaml
	$(ensure_out_dirs_exists)
	bb_train_decoder \
		--gt $(call setting,ground_truth,decoder.yaml) \
		--train-set $(call get,train_set,artificial_data_set) \
		--test-set $(call get,train_set,artificial_data_set) \
		--units $(call setting,units,decoder.yaml) \
		--epoch 1000 \
		--marker $(call setting,marker,decoder.yaml) \
		--model $(call setting,model,decoder.yaml) \
		--discriminator-threshold $(call setting,discriminator_threshold,decoder.yaml) \
		--make-json decoder \
		$(call setting,augmentations,decoder.yaml)   \
		$(models)

decoder_evaluation:
	bb_evaluate_decoder  $(call get,path,decoder)

decoder: decoder_model decoder_evaluation