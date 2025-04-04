clean:
	find app -type d -name '__pycache__' -exec rm -rf {} +
	find src -type d -name '__pycache__' -exec rm -rf {} +

compile: clean
	rm -rf tmp-codes
	mkdir -p tmp-codes
	cp -r app tmp-codes
	cp -r src tmp-codes
	cp app.py tmp-codes
