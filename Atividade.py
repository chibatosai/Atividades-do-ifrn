import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import OllamaEmbeddings

# Define os documentos
documents = [ "Construção de motores a combustão.", 
             "Tipos de ciclos de motores como otto, atikinson e diesel.", 
             "Motores turbos porque são mais potentes.", 
             "Os motores são composto por um Cabeçote, bloco do motor e o cárter e as móveis são: êmbolo ou pistão, camisas, conectora ou biela, virabrequim ou eixo de manivelas, eixo de cames, bomba de injeção.", 
             "Motores a combustão são feitos para converter energia quimica em trabalho e movimentar veículos em todo mundo."]

# Define o modelo de embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
document_embeddings = embeddings.embed_documents(documents)

# Mostra o tamanho dos embeddings
embedding_size = len(document_embeddings[0])
print(f"Tamanho dos embeddings: {embedding_size}")

# Realiza uma busca de similaridade para uma consulta dada
query = "quais as partes de um motor?"
query_embedding = embeddings.embed_query(query)

# Calcula os scores de similaridade
similarity_scores = cosine_similarity([query_embedding], document_embeddings)[0]

# Encontra o documento mais similar
most_similar_index = np.argmax(similarity_scores)
most_similar_document = documents[most_similar_index]

print(f"Documento mais similar à consulta '{query}':")
print(most_similar_document)