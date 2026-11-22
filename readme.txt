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
binario gerado sao ignorados pelo .gitignore. A conversao requer Python 3
e NumPy.

Linux:

  mkdir -p data
  wget -O data/covtype.data.gz https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz
  python3 scripts/prepare_dataset.py data/covtype.data.gz data/covtype.bin --drop-last

Windows PowerShell:

  New-Item -ItemType Directory -Force data
  py -m pip install numpy
  curl.exe -L https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz -o data/covtype.data.gz
  py scripts/prepare_dataset.py data/covtype.data.gz data/covtype.bin --drop-last

Formato do arquivo .bin:
  int32 N | int32 D | N*D float32 (ordem por linha)


3. ESTRUTURA DOS ARQUIVOS
---------------------------------------------------------------------
  kmeans_seq.c                versao sequencial
  kmeans_omp.c                versao paralela OpenMP (CPU multicore)
  kmeans_cuda.cu              versao paralela CUDA (GPU)
  scripts/prepare_dataset.py  conversao da base para o formato binario
  perguntas_entrevista.txt    roteiro de estudo para a entrevista
  data/                       dataset gerado localmente, nao versionado
  readme.txt                  este arquivo


4. COMPILACAO
---------------------------------------------------------------------
Pre-requisitos usados no Windows:
  - MSYS2 UCRT64 com GCC e suporte a OpenMP;
  - Visual Studio Build Tools 2022 com as ferramentas C++ para o nvcc;
  - NVIDIA CUDA Toolkit 13.5 e driver NVIDIA compativel.

Os comandos abaixo foram verificados no Windows 11. Antes de usar o GCC do
MSYS2, coloque seu diretorio no inicio do PATH para evitar que outra copia de
libwinpthread-1.dll seja carregada:

  $env:Path = "C:\msys64\ucrt64\bin;$env:Path"

Sequencial:
  gcc -O2 -std=c11 -Wall -o kmeans_seq.exe kmeans_seq.c -lm

OpenMP:
  gcc -O2 -std=c11 -Wall -fopenmp -o kmeans_omp.exe kmeans_omp.c -lm

CUDA (RTX 4060 Ti, arquitetura sm_89):
  O codigo usa clock_gettime para medir somente a leitura do arquivo. No
  Linux, compile diretamente:

  nvcc -O2 -arch=sm_89 -o kmeans_cuda kmeans_cuda.cu

  No Windows, execute no "x64 Native Tools Command Prompt for VS 2022". Como
  clock_gettime nao existe no MSVC, crie um cabecalho temporario, compile e
  remova-o; nenhuma funcao do programa precisa ser alterada:

  Conteudo de cuda_windows_compat.h:
    #define CLOCK_MONOTONIC TIME_UTC
    #define clock_gettime(clock_id, ts) timespec_get((ts), TIME_UTC)

  "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v13.5\bin\nvcc.exe" -O2 -arch=sm_89 --pre-include cuda_windows_compat.h -o kmeans_cuda.exe kmeans_cuda.cu
  del cuda_windows_compat.h


5. EXECUCAO
---------------------------------------------------------------------
Parametros: <dataset.bin> <K> <max_iter> [arquivo_rotulos.txt]
  K                  numero de clusters
  max_iter           limite de iteracoes
  arquivo_rotulos    opcional; grava um rotulo por ponto

Parametros usados nas medicoes: K = 32, max_iter = 300.

Sequencial (PowerShell):
  .\kmeans_seq.exe .\data\covtype.bin 32 300

OpenMP (o numero de threads e definido por OMP_NUM_THREADS):
  $env:OMP_NUM_THREADS=1;  .\kmeans_omp.exe .\data\covtype.bin 32 300
  $env:OMP_NUM_THREADS=2;  .\kmeans_omp.exe .\data\covtype.bin 32 300
  $env:OMP_NUM_THREADS=4;  .\kmeans_omp.exe .\data\covtype.bin 32 300
  $env:OMP_NUM_THREADS=8;  .\kmeans_omp.exe .\data\covtype.bin 32 300
  $env:OMP_NUM_THREADS=16; .\kmeans_omp.exe .\data\covtype.bin 32 300
  $env:OMP_NUM_THREADS=32; .\kmeans_omp.exe .\data\covtype.bin 32 300

CUDA:
  .\kmeans_cuda.exe .\data\covtype.bin 32 300


6. AMBIENTE E RESULTADOS
---------------------------------------------------------------------
Ambiente da medicao:
  Sistema: Windows 11 Pro 64 bits (10.0.26200)
  CPU: AMD Ryzen 7 5700X3D, 8 nucleos e 16 threads logicas
  GPU: NVIDIA GeForce RTX 4060 Ti, 8 GB
  Driver NVIDIA: 610.88
  CUDA Toolkit: 13.5
  Arquitetura CUDA compilada: sm_89
  GCC/MSYS2: 16.1.0
  MSVC: 19.44.35230
  Plano de energia: Equilibrado

Dataset e parametros da medicao:
  Dataset: Covertype convertido para data/covtype.bin
  N = 581012, D = 54, K = 32, max_iter = 300

Metodo de medicao:
  - apenas o laco do K-means foi cronometrado;
  - leitura, inicializacao, SSE e copia inicial para a GPU ficaram de fora;
  - uma execucao de aquecimento foi descartada por configuracao;
  - foram feitas 5 execucoes medidas por configuracao;
  - todas as configuracoes e rodadas foram aceitas somente apos conferir
    iteracoes, SSE e tamanhos dos clusters contra a versao sequencial;
  - a tabela apresenta a mediana das 5 execucoes;
  - todos os speedups usam a mediana sequencial como referencia.

Resultados obtidos na mesma maquina:

  Versao                Tempo K-means (s)    Speedup
  Sequencial            16.180               1.00
  OpenMP  1 thread      16.974               0.95
  OpenMP  2 threads       8.619               1.88
  OpenMP  4 threads       4.406               3.67
  OpenMP  8 threads       2.536               6.38
  OpenMP 16 threads       1.744               9.28
  OpenMP 32 threads       1.738               9.31
  CUDA                    0.614              26.35

Speedup = tempo sequencial / tempo da versao paralela.

A CPU possui 16 threads logicas. Portanto, a configuracao OpenMP com 32
threads usa sobrescricao: ela foi medida para atender ao enunciado, mas nao
representa 32 nucleos ou 32 threads de hardware. A diferenca pequena entre
16 e 32 threads deve ser interpretada considerando essa limitacao.

Validacao sequencial x OpenMP x CUDA:
  Iteracoes: 34 (convergiu)
  SSE: 1.440870e+07
  Tamanhos de cluster identicos nas tres versoes:
  1899 2845 8750 16504 37142 9259 14280 7273
  1147 78668 9700 15573 6450 29969 11633 6452
  77593 33191 14720 4021 14033 21278 17431 12515
  2032 2729 2589 9645 46767 7591 52510 4823

A versao sequencial levou mais de 10 segundos na base real, atendendo ao
requisito da especificacao. As tres versoes convergiram com o mesmo SSE e a
mesma distribuicao de pontos entre os clusters.


7. SITUACAO DA IMPLEMENTACAO
---------------------------------------------------------------------
  Sequencial: implementada e validada.
  OpenMP: implementada e validada para 1, 2, 4, 8, 16 e 32 threads.
  CUDA: implementada e validada na NVIDIA GeForce RTX 4060 Ti.
