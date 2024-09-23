.PHONY: nbconvert

nbconvert:
	jupyter nbconvert --to markdown ./notebooks/supervised_training.ipynb --output-dir=./docs

	
VERSION=$(shell python setup.py --version)

.PHONY: version

version:
	@NEW_VERSION=$$(echo $(VERSION) | awk -F. '{print $$1"."$$2"."$$3+1}'); \
	echo "new version: $$NEW_VERSION"; \
	echo "exec replacement s/version='$(VERSION)'/version='$$NEW_VERSION'/g in setup.py"; \
	sed -i "s/version='$(VERSION)'/version='$$NEW_VERSION'/g" setup.py; \
	git add setup.py; \
	git commit -m "version $$NEW_VERSION"; \
	git push; \
	git tag v$$NEW_VERSION; \
	git push --tags 


.PHONY: paradoc

paradoc:
	python doc_tools.py tabular_transformer/hyperparameters.py > docs/hyperparameters.md




# Run a specific notebook
run:
	@NOTEBOOK=$(word 2,$(MAKECMDGOALS)); \
	if [ "$$NOTEBOOK" = "" ]; then \
		echo "Usage: make run notebook.ipynb"; \
		exit 1; \
	else \
		echo "Executing $$NOTEBOOK..."; \
		jupyter nbconvert --to notebook --stdout --execute "$$NOTEBOOK"; \
	fi