=====================================================================
 K-MEANS SEQUENCIAL, OPENMP E CUDA
 Trabalho Prático - Computação Paralela - PUC Minas
 Integrantes: Alessandra Faria / Débora Luiza / Douglas Nícolas
=====================================================================

1. SOBRE A APLICAÇÃO
---------------------------------------------------------------------
O K-means é um algoritmo de agrupamento (clustering) não supervisionado:
recebe N pontos com D atributos e os divide em K grupos (clusters), de
forma que cada ponto fique no grupo do centróide mais próximo.

Algoritmo (Lloyd):
  1. Inicialização: escolhe K centróides iniciais. Usamos uma escolha
     determinística (centróide c = ponto c * N/K), para que todas as
     versões partam dos mesmos centróides e possam ser comparadas.
  2. Atribuição: cada ponto é associado ao centróide mais próximo
     (distância euclidiana ao quadrado). Custo O(N*K*D).
  3. Atualização: cada centróide passa a ser a média dos pontos do seu
     cluster. Custo O(N*D).
  4. Repete os passos 2 e 3 até nenhum ponto trocar de cluster
     (convergência) ou até atingir max_iter.

Ao final, o programa imprime o número de iterações, o SSE (soma das
distâncias ao quadrado de cada ponto até seu centróide) e o tamanho de
cada cluster. Esses valores servem para validar as versões paralelas:
elas devem chegar ao mesmo número de iterações e ao mesmo SSE (com
diferença apenas nas últimas casas decimais, por causa da ordem das
somas em ponto flutuante).

O tempo medido é apenas o do laço do K-means (passos 2 a 4); a leitura
do arquivo não entra na medição.


2. BASE DE DADOS
---------------------------------------------------------------------
Covertype - UCI Machine Learning Repository
  581.012 pontos x 54 atributos (dados cartográficos de áreas florestais)
  https://archive.ics.uci.edu/ml/datasets/Covertype

A última coluna do arquivo original (classe do tipo de cobertura) é
descartada, já que o K-means é não supervisionado. Os atributos são
normalizados por z-score (média 0, desvio padrão 1) para que atributos
com escala grande (ex.: elevação em metros) não dominem a distância.

Download e conversão para o formato binário usado pelos programas
(requer Python 3 com NumPy):

  mkdir -p data
  curl -L -o data/covtype.data.gz https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz
  python3 scripts/prepare_dataset.py data/covtype.data.gz data/covtype.bin --drop-last

Formato do arquivo .bin: int32 N | int32 D | N*D float32 (linha a linha).


3. ESTRUTURA DOS ARQUIVOS
---------------------------------------------------------------------
  kmeans_seq.c                versão sequencial
  kmeans_omp.c                versão paralela OpenMP (CPU multicore)
  kmeans_cuda.cu              versão paralela CUDA (GPU)
  scripts/prepare_dataset.py  download/conversão da base para .bin
  data/                       base de dados (covtype.bin)
  readme.txt                  este arquivo


4. COMPILAÇÃO
---------------------------------------------------------------------
Sequencial:
  gcc -O2 -std=c11 -Wall -o kmeans_seq kmeans_seq.c -lm

OpenMP:
  gcc -O2 -std=c11 -Wall -fopenmp -o kmeans_omp kmeans_omp.c -lm

CUDA:
  nvcc -O2 -o kmeans_cuda kmeans_cuda.cu

Observação (Windows/MSYS2): se o gcc acusar "undefined reference to
clock_gettime", acrescente -lpthread ao final do comando.


5. EXECUÇÃO
---------------------------------------------------------------------
Parâmetros: <dataset.bin> <K> <max_iter> [arquivo_rotulos.txt]
  K                  número de clusters
  max_iter           limite de iterações (o algoritmo para antes se convergir)
  arquivo_rotulos    opcional; grava o cluster de cada ponto, um por linha

Parâmetros usados nas medições: K = 32, max_iter = 300

Sequencial:
  ./kmeans_seq data/covtype.bin 32 300

OpenMP (o número de threads é definido por OMP_NUM_THREADS):
  OMP_NUM_THREADS=1  ./kmeans_omp data/covtype.bin 32 300
  OMP_NUM_THREADS=2  ./kmeans_omp data/covtype.bin 32 300
  OMP_NUM_THREADS=4  ./kmeans_omp data/covtype.bin 32 300
  OMP_NUM_THREADS=8  ./kmeans_omp data/covtype.bin 32 300
  OMP_NUM_THREADS=16 ./kmeans_omp data/covtype.bin 32 300
  OMP_NUM_THREADS=32 ./kmeans_omp data/covtype.bin 32 300

CUDA:
  ./kmeans_cuda data/covtype.bin 32 300


6. RESULTADOS
---------------------------------------------------------------------
Máquina de teste: ______________________ (CPU, nº de núcleos, GPU)
Resultado de referência (sequencial): 34 iterações, SSE = 1.440870e+07

  Versão                Tempo (s)    Speedup
  Sequencial            16.150       1.00
  OpenMP  1 thread      ______       ____
  OpenMP  2 threads     ______       ____
  OpenMP  4 threads     ______       ____
  OpenMP  8 threads     ______       ____
  OpenMP 16 threads     ______       ____
  OpenMP 32 threads     ______       ____
  CUDA                  ______       ____

Speedup = tempo sequencial / tempo da versão paralela.