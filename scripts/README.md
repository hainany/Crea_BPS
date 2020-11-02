This is the scripts dir of the project "Bad Poets Society".

The main script is the 'poetrylstm.py' file.
To run it, please type 
	python3 poetrylstm.py 
in your command line.  

## Content
Besides 'poetrylstm.py', this dir also contains the training data (a.k.a. inspiring set, see dir 'data') and two trained models used in this system:  

* trained Word2vec model
* trained LSTM model  

Please uncomment the relative lines in the 'poetrylstm.py' script to load these models.  


## Environment requirements

python>=3.9.6
keras>=2.4.3
tensorflow>=2.3.1
pandas>=1.1.4
gensim>=3.8.3
numpy>=1.18.5