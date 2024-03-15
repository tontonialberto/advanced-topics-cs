FROM python:3.10-slim

ADD docker-entrypoint.sh /

ADD requirements.txt /

RUN pip install --no-cache-dir -r requirements.txt

ADD src /src

ADD resources /resources

ENTRYPOINT [ "/docker-entrypoint.sh" ]