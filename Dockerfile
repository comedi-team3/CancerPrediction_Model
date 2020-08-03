FROM ubuntu:18.04
FROM python:3

MAINTAINER lydiahjchung "lydiahjchung@gmail.com"

WORKDIR /workspace

RUN apt-get update -y
RUN apt-get upgrade -y

COPY requirements.txt ./

RUN apt install vim -y
RUN pip install -r requirements.txt


#CMD ["sleep", "3600"]
