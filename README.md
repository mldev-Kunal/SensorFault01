# Sensor Fault Detection(Wafer production)

## 🚀 Deployment
[![Deployed on AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white)]

**Access the live application for new predictions here:** [Wafer Fault Detection_Tool](http://13.61.16.83:8080/predict)

**For home page - http://13.61.16.83:8080 **

## Background

In semiconductor manufacturing, a wafer (also referred to as a substrate) is a thin slice of crystalline silicon (c-Si) that serves as the foundation for integrated circuits and photovoltaic (PV) cells. The fabrication process involves complex microfabrication stages, including:

Doping & Ion Implantation: Modifying electrical properties.

Etching & Deposition: Removing or adding thin-film layers.

Photolithography: Patterning the micro-circuitry.

Once processed, wafers are diced into individual chips or cells and packaged for final use.

## Problem Statement

Modern photovoltaic systems rely on massive arrays of wafers located in remote, high-scale facilities. Each wafer is equipped with hundreds of sensors monitoring its performance and structural integrity.

## Challenges

Manual Inspection Inefficiency: Detecting faulty wafers manually is labor-intensive, prone to error, and difficult to scale in remote locations.

Operational Downtime: Physical inspection requires dismantling components and halting production for surrounding wafers. If a suspicion leads to a false negative (misidentifying a functional wafer as faulty), the resulting downtime leads to significant revenue loss and wasted man-hours.

High Stakes: Because photovoltaic power generation requires precision technology, even minor defects can drastically reduce energy conversion efficiency across the entire system.

## Solution Proposed
This project implements an end-to-end Machine Learning pipeline designed to process high-dimensional sensor data directly from the wafers. Data fetched by wafers is to be passed through the machine learning pipeline and it is to be determined whether the wafer at hand is faulty or not apparently obliterating the need and thus cost of hiring manual labour.

### By leveraging predictive modeling, we can determine the health of a wafer in real-time without physical intervention. This solution:

Eliminates the need for constant manual monitoring.

Reduces unnecessary operational shutdowns by providing high-precision defect detection.

Optimizes maintenance costs and improves the overall reliability of the photovoltaic power generation system.
