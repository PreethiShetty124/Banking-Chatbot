FROM python:3.10.7

RUN pip3 install nltk
RUN python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"


#copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt


#switch working directory
WORKDIR /app

# copy every content from the local file to the image
COPY . /app

# install the dependencies and packages in the requirements file
RUN pip install -r requirements.txt

EXPOSE 5000

# define the command to start the container
CMD [ "python","./chatbot app.py" ]
