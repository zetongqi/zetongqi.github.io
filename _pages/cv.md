---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

Education
======
* B.S. in Electrical Engineering, Shandong University, 2017
* M.S. in Electrical and Computer Engineering, University of Wisconsin-Madison, 2019
* Ph.D in Computer Science, University of Illinois at Chicago, 2025 (expected)

Work experience
======
* Teaching Assistant
  * University of Wisconsin-Madison
  * TA for CS 252: Introduction to Computer Engineering and ECE 271: Circuits Laboratory II. Basic responsibilities include designing and grading homework and exams, leading discussion sessions, tutorial sessions, and lab sessions, answering Piazza questions, hosting office hours weekly, and proctoring exams
  * Supervisor: Adil Ibrahim

* Machine Learning Research Intern
  * American Family Insurance
  * Developed the IGLUE benchmark to create a reliable metric for language models’ performance in insurance literature. Trained language models for retrieval-based chatbots to bring better customer service. Developed multitask training methods to create language models with significantly better performance. Created a language model with 0.6 percent boost on IGLUE score by pretraining Google’s BERT model with claims notes data. Created quality datasets from raw text data with Amazon Athena for fine-tuning language models.
  * Supervisor: Glenn Fung
  
* Data Analyst
  * Zeno Group
  * Implemented data visualizations using Python and D3 to create intuitive, easy to understand, and interactive topline reports. Unbiasing data to create more general and representative datasets. Cleaning raw data with Python to create quality datasets
  
Skills
======
* Machine Learning
* Optimization
* Python

Publications
======
  <ul>{% for post in site.publications %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  
Talks
======
  <ul>{% for post in site.talks %}
    {% include archive-single-talk-cv.html %}
  {% endfor %}</ul>
  
Teaching
======
  <ul>{% for post in site.teaching %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  
