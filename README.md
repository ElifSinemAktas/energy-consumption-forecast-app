### In progress
# Energy Consumption Forecast App

This is the final step/project of my journey through MLOps bootcamp, where I've had the opportunity to apply cutting-edge technologies like FastAPI, MLflow, Jenkins, Terraform etc and tap into real-time data from EPIAS. This project has been a labor of love, and it's designed to provide an end-to-end solution for monitoring and analyzing energy consumption data, sourced directly from EPIAS's Real Time Consumption data.

## 1. Project Overview
Energy consumption analysis is not just a practical skill; it's a critical aspect of sustainability and resource management. Here's a personal touch on how my project addresses these requirements:

### 1.1. Dataset Integration
I fetch and process real-time energy consumption data from EPIAS, ensuring that my analysis is always based on the latest information available.

### 1.2 Modeling
My project includes machine learning model that I developed to achieve these specific goals:

#### 1.2.1. Long-term Forecasting: 
I've built an ML model that predicts electricity consumption for the next 5 days.

#### 1.2.2. Short-term Forecasting: 
There's also an ML model in place for predicting electricity consumption over the next 24 hours.

#### 1.2.3. ML Pipeline: 
To streamline my modeling process, I've implemented an ML pipeline that enhances efficiency and model management.

### 1.3. Deployment
My plain API offers the following functionalities:

#### 1.3.1. Daily and Hourly Forecasts: 
The API provides separate endpoints for daily and hourly electricity consumption forecasts.

#### 1.3.2. Flexible Parameters: 
Users can specify the date, number of days, and time as path or query parameters to tailor their consumption predictions.

#### 1.3.3. Customized Results: 
As a result, the API delivers estimated electricity consumption for the requested day(s) or hour(s).

#### 1.3.4. Model Concept/Data Drift Detection: 
I've incorporated a mechanism for detecting any drift in the model's concept or data, ensuring that my predictions remain reliable over time.

#### 1.3.5. Automated Deployment: 
Jenkins/Gitea handles model deployment, streamlining the process and ensuring that updated models are readily available for use.

### 1.4. Infrastructure
I've built my project on a solid infrastructure foundation:

#### 1.4.1. Docker Containers: 
My infrastructure is containerized using Docker, making it easy to deploy and scale the project as needed.

#### 1.4.2. MySQL Database: 
I utilize MySQL as my database system, storing estimation results for future reference and analysis.

#### 1.4.3. Data Persistence: 
The project writes estimation results into the MySQL database for data retention and analysis purposes.

## 2. Getting Started
To begin exploring my Energy Consumption Project, follow the installation instructions in the Installation section below. Whether you want to set up your own instance or simply explore the project, I provide detailed guidance to help you get started.

### 2.1. Architecture
![Simple Architecture](./images/architecture.png)

### 2.2 Installation