
USER := $(shell whoami)
.PHONY: rendergan

all: base-image custom-image

base-image:
	docker build -t rendergan .

custom-image:
	./generate_user_dockerfile.sh
	cd rendergan-$(USER)
	docker build --tag rendergan-$(USER) .
