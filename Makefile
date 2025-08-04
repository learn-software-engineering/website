.ONESHELL:
MAKEFLAGS += --no-builtin-rules

help:
	@printf "Usage: make [target] [VARIABLE=value]\nTargets:\n"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install: ## Install pre-commit hooks
	@pre-commit install
	@pre-commit gc

uninstall: ## Uninstall hooks
	@pre-commit uninstall

validate: ## Validate files with pre-commit hooks
	@pre-commit run --all-files

spell-check: ## Run spell check on markdown files
	@cspell --config .github/workflows/scripts/cspell-config-en.json "content/**/*.en.md"
	# @cspell --config .github/workflows/scripts/cspell-config-es.json "content/**/*.es.md"

build: ## Builds the website package
	@ENV=$(env) .github/workflows/scripts/build.sh

run: ## Start local server. Receives `env` argument. `env=production` will run in production mode. Any other value will fallback to development
	@if [ "$(env)" = "production" ]; then \
		echo "### Running in production mode ###"; \
		ENV_SETTINGS="--gc --minify --templateMetrics"; \
	else \
		echo "### Running in development mode ###"; \
		ENV_SETTINGS="--buildDrafts --buildExpired --buildFuture --disableFastRender --navigateToChanged"; \
	fi; \
	hugo server --bind 0.0.0.0 --environment $(env) $$ENV_SETTINGS \
		--printI18nWarnings --printMemoryUsage --printPathWarnings --printUnusedTemplates

update: ## Updates project dependencies
	@hugo mod get -u ./...
	@hugo mod tidy
