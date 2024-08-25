
# Housing Prices ML

  

## Installation

  

To set up the project, follow these steps:

  

1.  **Install the required packages**

	Install all the necessary packages by running the following command in your terminal:

  

```bash
	pip install -r requirements.txt
```
  

2.  **Run the server**

	After installing the required packages, start the server by running:

```bash
	python server.py
```
  

## Installation with Docker

1.  **Build the Docker Image**

	To install the project using Docker, run the following command in your terminal:

```bash
	docker build -t housing-prices-prediction .
```
  
2.  **Run the application container**

	Run the server by running the following command in your terminal:

```bash
	docker run -p 8000:8000 housing-prices-prediction
```

## Get predictions

  The file `api_request_predict.http` has a REST POST example to use against the server to get housing prices predictions