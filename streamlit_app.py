import streamlit as st
from joblib import dump, load
import numpy as np
import os


# streamlit_app = nome do app

st.header('Aplicação')
st.subheader('Construído em Python')
st.markdown('insira as informações para efetuar as previsões')

 #exemplo texto
nome = st.text_input('Informe o nome', max_chars=30)
st.write(nome)

 #exemplo de número
numero = st.number_input('Informe um número')
st.write(numero)


if (os.path.exists('extra_trees_regressor.pkl')):
    modelo = load('extra_trees_semcolunempresa.pkl')
    botao = st.button('Efetuar previsão')
    if(botao):
    	listaValores = np.array([[nome, numero]])
    	resultado = modelo.predict(listaValores)
    	if(resultado[0] == 0):
    		st.write('Setosa')
    	elif (resultado[0] == 1):
    		st.write('Versicolour')
    	else:
    		st.write('Virginica')
else:
 st.error('Erro ao carregar o modelo preditivo. Contrate o administrador do sistema')
