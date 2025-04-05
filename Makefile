copy:
	@rm -rf tmp
	@mkdir -p tmp
	@rsync -a \
		--exclude='.git' \
		--exclude='.venv' \
		--exclude='logs' \
		--exclude='outputs' \
		--exclude='__pycache__' \
		--exclude='*tmp*' \
		--exclude='tmp' \
		. tmp/
	@echo "Project copied to tmp/"

deploy:
	@streamlit run app.py
