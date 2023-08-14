FROM jenkins/jenkins:2.342-jdk11

USER root

# Install ansible
RUN apt update && \
apt -y install python3-distutils && \
apt -y install python3-apt && \
apt -y install python3

RUN curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py" && python3 get-pip.py

RUN pip install -U ansible

RUN apt-get -y install rsync

USER jenkins

