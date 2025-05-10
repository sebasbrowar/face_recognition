import pickle
import os
import logging

with open('rostros.pkl', 'rb') as f:
    known_faces = pickle.load(f)

print(known_faces)  # Imprime el contenido de rostros.pkl

