.PHONY: install test run docker
install:
	python -m pip install -r requirements.txt

test:
	pytest

run:
	streamlit run app.py

docker:
	docker compose up --build
