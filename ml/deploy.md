# Deployment

<!-- MarkdownTOC -->

- Overview
- Deploying, Serving, and Inferencing Models at Scale
- Observing and Monitoring Model in Production
- Deployment Examples
- Deployment Franeworks
    - Fastapi
    - gRPC
- Cloud Services
    - Streamlit
    - Deta
- References

<!-- /MarkdownTOC -->

## Overview

Consider using model life cycle development and management platforms like MLflow, DVC, Weights & Biases, or SageMaker Studio. And Ray, Ray Tune, Ray Train (formerly Ray SGD), PyTorch and TensorFlow for distributed, compute-intensive and deep learning ML workloads.

NOTE: Consider feature stores as part of your model development process. Look to Feast, Tecton, SageMaker, and Databricks for feature stores. 


## Deploying, Serving, and Inferencing Models at Scale

Once the model is trained and tested and meets the business requirements for model accuracy, there are seven crucial requirements for scalable model serving frameworks to consider are:

**Framework agnostic:** A model serving framework should be ML framework agnostic, so it can deploy any common model built with common ML frameworks such as PyTorch, TensorFlow, XGBoost, or Scikit-learn, each with its own algorithms and model architectures.

**Business Logic:** Model prediction often requires preprocessing, post processing or ability to augment request data by connecting to a feature store or any other data store for validation. Model serving should allow this as part of its inference.

**Model Replication:** Some models are compute-intensive or network-bound. Therefor, the framework should be able to fan out requests to model replicas, load balancing among replicas to support parallel request handling during peak traffic.

**Request Batching:** Not all models in production are employed for real-time serving. Often, models are scored in large batches of requests. For deep learning models, parallelizing image requests to multiple cores and taking advantage of hardware accelerators to expedite batch scoring and utilize hardware resources.

**High Concurrency and Low Latency:** Models in production require real-time inference with low latency while handling bursts of heavy traffic of requests which is crucial for best user experience to receive millisecond responses on prediction requests.

**Model Deployment CLI and APIs:** A ML engineer responsible for deploying a model should be able to easily use model server’s deployment APIs or command line interfaces (CLI) to deploy model artifacts into production which allows model deployment from within an existing CI/CD pipeline or workflow.

**Patterns of Models in Production:** As ML applications are increasingly becoming pervasive in all sectors of industry, models trained for these ML applications are complex and composite. 

Thus, models do not exist in isolation and they do not predict results singularly. They operate jointly and often in four model patterns: pipeline, ensemble, business logic, and online learning. Each pattern has its purpose and merits.

Machine Learning engineers adopt two common approaches to deploy these patterns of models in production: embed models into a web server and offload to an external service. Each approach has its own pros and cons.

NOTE: Look to Seldon, KFServing, or Ray Serve for all these seven requirements.

## Observing and Monitoring Model in Production

Model monitoring is critical to model viability in the post deployment production stage whoch is often overlooked. 

**Data drift over time:** The quality and accuracy of the model depends on the quality of the data which is complex and never static. The original model was trained with the extracted features may not be as important over time. Some new features may emerge that need to be taken into account. Such features drifts in data require retraining and redeploying the model because the distribution of the variables is no longer relevant.

**Model concept changes over time:** Many practitioners refer to this as model decay or model staleness. When the patterns of trained models no longer hold with the drifting data, the model is no longer valid because the relationships of its input features may not necessarily produce the expected prediction. This, model accuracy degrades.

**Models fail over time:** Models fail for inexplicable reasons: a system failure or bad network connection; an overloaded system; a bad input or corrupted request. Detecting these failures root causes early or its frequency mitigates bad user experience and deters mistrust in the service if the user receives wrong or bogus outcomes.

**Systems degrade over load:** Constantly being vigilant of the health of dedicated model servers or services deployed is also important: data stores, web servers, routers, cluster nodes’ system health, etc.

Collectively, these aforementioned monitoring model concepts are called _model observability_ which is important in MLOps best practices. Monitoring the health of data and models should be part of the model development cycle.

NOTE: For model observability look to Evidently.ai, Arize.ai, Arthur.ai, Fiddler.ai, Valohai.com, or whylabs.ai.



## Deployment Examples

[Serving ML Models with gRPC](https://towardsdatascience.com/serving-ml-models-with-grpc-2116cf8374dd)

[The Nice Way To Deploy An ML Model Using Docker](https://towardsdatascience.com/the-nice-way-to-deploy-an-ml-model-using-docker-91995f072fe8)

[Deploying Your First Machine Learning API using FastAPI and Deta](https://www.kdnuggets.com/2021/10/deploying-first-machine-learning-api.html)

[Deploy MNIST Trained Model as a Web Service using Flask](https://towardsdatascience.com/deploy-mnist-trained-model-as-a-web-service-ba333d233a5d)



## Deployment Franeworks

### Fastapi

Similar to the flask, fastapi is also a popular framework in Python for web backend development. 

Fastapi focuses on using the least code to write regular Web APIs which is good if backend is not too complex. 


### gRPC

The trend now is toward gRPC for microservices since it is more secure, faster, and more robust (especially with IoT).

[gRPC with REST and Open APIs](https://grpc.io/blog/coreos/)

- gRPC was recommended for developing microservices by my professor in Distributed Computing course.

- gRPC uses HTTP/2 which enables applications to present both a HTTP 1.1 REST/JSON API and an efficient gRPC interface on a single TCP port (available for Go). 

- gRPC provides developers with compatibility with the REST web ecosystem while advancing a new, high-efficiency RPC protocol. 

- With the recent release of Go 1.6, Go ships with a stable net/http2 package by default.



## Cloud Services

### [Streamlit](https://streamlit.io/)

Streamlit is a Python package that makes it very easy to create dashboards and data applications without the need for any front-end programming expertise

All in Python. All for free. No front‑end experience required.

### [Deta](https://www.deta.sh)

Deta is a free, developer friendly cloud platform. 

**Deta Micros** is a service to deploy Python and Node.js apps/APIs on the internet in seconds. 

**Deta Base** is a super easy to use production-grade NoSQL database that comes with unlimited storage.

**Deta Drive** is an easy to use cloud storage solution by Deta – get 10GB for free



## References

[Considerations for Deploying Machine Learning Models in Production](https://towardsdatascience.com/considerations-for-deploying-machine-learning-models-in-production-89d38d96cc23?source=rss----7f60cf5620c9---4)

