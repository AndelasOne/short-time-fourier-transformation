FROM ubuntu:latest

RUN apt-get update && apt-get -y update
RUN apt-get install -y build-essential python3.6 python3-pip python3-dev
RUN apt-get install -y libportaudio2
RUN apt-get install -y libsndfile1
RUN pip3 -q install pip --upgrade



ENV APP_HOME /app
WORKDIR ${APP_HOME}

COPY . ./


RUN pip install pip pipenv --upgrade
RUN pipenv install --skip-lock --system --dev


CMD ["sh","./scripts/entrypoint.sh"]