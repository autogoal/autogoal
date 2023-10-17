
build:
	docker build --file ./dockerfile -t autogoal:production .
	docker save -o autogoal-prod.tar autogoal:production

serve: build
	docker run -p 8000:8000 autogoal:production

        