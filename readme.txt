=====================================================================
 K-MEANS SEQUENCIAL, OPENMP E CUDA
 Trabalho Pratico - Computacao Paralela - PUC Minas
 Integrantes: Alessandra Faria / Debora Luiza / Douglas Nicolas
=====================================================================

1. SOBRE A APLICACAO
---------------------------------------------------------------------
O K-means e um algoritmo de agrupamento (clustering) nao supervisionado:
recebe N pontos com D atributos e os divide em K grupos (clusters), de
forma que cada ponto fique no grupo do centroide mais proximo.

Algoritmo de Lloyd:
  1. Inicializacao: escolhe K centroides iniciais. Usamos uma escolha
     deterministica (centroide c = ponto c * N/K), para que todas as
     versoes partam dos mesmos centroides e possam ser comparadas.
  2. Atribuicao: cada ponto e associado ao centroide mais proximo
     (distancia euclidiana ao quadrado). Custo O(N*K*D).
  3. Atualizacao: cada centroide passa a ser a media dos pontos do seu
     cluster. Custo O(N*D).
  4. Repete os passos 2 e 3 ate nenhum ponto trocar de cluster
     (convergencia) ou ate atingir max_iter.

Ao final, o programa imprime o numero de iteracoes, o SSE (soma das
distancias ao quadrado de cada ponto ate seu centroide) e o tamanho de
cada cluster. Esses valores servem para validar as versoes paralelas.

O tempo principal medido e somente o laco do K-means (passos 2 a 4); a
leitura do arquivo nao entra na medicao. Na versao CUDA, a copia inicial
dos dados do host para a GPU tambem fica fora do evento de medicao, para
comparar o mesmo laco computacional entre as versoes.


2. BASE DE DADOS
---------------------------------------------------------------------
Dataset utilizado: Covertype - UCI Machine Learning Repository
  581.012 pontos x 54 atributos
  https://archive.ics.uci.edu/ml/datasets/Covertype

A ultima coluna do arquivo original (classe do tipo de cobertura) e
descartada, ja que o K-means e nao supervisionado. Os 54 atributos sao
normalizados por z-score (media 0, desvio padrao 1) para evitar que
atributos com escala maior dominem a distancia euclidiana.

Os arquivos de dados nao sao versionados no Git. O arquivo bruto e o
binario gerado sao ignorados pelo .gitignore, e podem ser reproduzidos
com os comandos abaixo (requer Python 3, NumPy e wget ou curl):

  mkdir -p data
  wget -O data/covtype.data.gz https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz
  python3 scripts/prepare_dataset.py data/covtype.data.gz data/covtype.bin --drop-last

Formato do arquivo .bin:
  int32 N | int32 D | N*D float32 (ordem por linha)


3. ESTRUTURA DOS ARQUIVOS
---------------------------------------------------------------------
  kmeans_seq.c                versao sequencial
  kmeans_omp.c                versao paralela OpenMP (CPU multicore)
  kmeans_cuda.cu              versao paralela CUDA (GPU)
  scripts/prepare_dataset.py  conversao da base para o formato binario
  data/                       dataset gerado localmente, nao versionado
  readme.txt                  este arquivo


4. COMPILACAO
---------------------------------------------------------------------
Os comandos abaixo foram executados no Ubuntu dentro do WSL 2.

Sequencial:
  gcc -O2 -std=c11 -Wall -o kmeans_seq kmeans_seq.c -lm

OpenMP:
  gcc -O2 -std=c11 -Wall -fopenmp -o kmeans_omp kmeans_omp.c -lm

CUDA (RTX 3050, arquitetura sm_86):
  nvcc -O2 -arch=sm_86 -o kmeans_cuda kmeans_cuda.cu


5. EXECUCAO
---------------------------------------------------------------------
Parametros: <dataset.bin> <K> <max_iter> [arquivo_rotulos.txt]
  K                  numero de clusters
  max_iter           limite de iteracoes
  arquivo_rotulos    opcional; grava um rotulo por ponto

Parametros usados nas medicoes: K = 32, max_iter = 300.

Sequencial:
  ./kmeans_seq data/covtype.bin 32 300

OpenMP (o numero de threads e definido por OMP_NUM_THREADS):
  OMP_NUM_THREADS=1  ./kmeans_omp data/covtype.bin 32 300
  OMP_NUM_THREADS=2  ./kmeans_omp data/covtype.bin 32 300
  OMP_NUM_THREADS=4  ./kmeans_omp data/covtype.bin 32 300
  OMP_NUM_THREADS=8  ./kmeans_omp data/covtype.bin 32 300
  OMP_NUM_THREADS=16 ./kmeans_omp data/covtype.bin 32 300
  OMP_NUM_THREADS=32 ./kmeans_omp data/covtype.bin 32 300

CUDA:
  ./kmeans_cuda data/covtype.bin 32 300


6. AMBIENTE E RESULTADOS
---------------------------------------------------------------------
Ambiente da medicao:
  Sistema: WSL 2 / Ubuntu
  GPU: NVIDIA GeForce RTX 3050
  Driver NVIDIA no Windows: 610.88
  CUDA Toolkit: 13.5
  Arquitetura CUDA compilada: sm_86

Dataset e parametros da medicao:
  Dataset: Covertype convertido para data/covtype.bin
  N = 581012, D = 54, K = 32, max_iter = 300

Resultados obtidos:

  Versao                Tempo K-means (s)    Speedup
  Sequencial            25.939               1.00
  OpenMP  1 thread       ______               ____
  OpenMP  2 threads      ______               ____
  OpenMP  4 threads      ______               ____
  OpenMP  8 threads      ______               ____
  OpenMP 16 threads      ______               ____
  OpenMP 32 threads      ______               ____
  CUDA                   2.685               9.66

Speedup = tempo sequencial / tempo da versao paralela.

Validacao sequencial x CUDA:
  Iteracoes: 34 (convergiu)
  SSE: 1.440870e+07
  Tamanhos de cluster identicos nas duas versoes:
  1899 2845 8750 16504 37142 9259 14280 7273
  1147 78668 9700 15573 6450 29969 11633 6452
  77593 33191 14720 4021 14033 21278 17431 12515
  2032 2729 2589 9645 46767 7591 52510 4823

Tempos de leitura observados, excluidos do speedup:
  Sequencial: 0.473 s
  CUDA:      0.464 s

A versao sequencial levou mais de 10 segundos na base real, atendendo ao
requisito da especificacao. As medicoes de OpenMP ainda precisam ser
preenchidas depois da implementacao e execucao da versao OpenMP.


7. SITUACAO DA IMPLEMENTACAO
---------------------------------------------------------------------
  Sequencial: implementada e validada.
  CUDA: implementada e validada na NVIDIA GeForce RTX 3050.
  OpenMP: ainda pendente.
