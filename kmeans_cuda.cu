/*
 * =====================================================================
 *  K-MEANS - VERSÃO SEQUENCIAL
 *  Trabalho Prático - Computação Paralela - PUC Minas
 *  Integrantes: Alessandra Faria / Débora Luiza / Douglas Nícolas
 * =====================================================================
 *
 *  Base de dados: Covertype (UCI) - 581.012 pontos x 54 atributos
 *  Parâmetros: K = 32, max_iter = 300
 *  Resultado de referência: 34 iterações, SSE = 1.440870e+07
 *  Máquina de teste: ______________________
 *
 *  TEMPOS DE EXECUÇÃO (apenas o laço do K-means, sem leitura do arquivo)
 *  -------------------------------------------------------------------
 *   Sequencial ............ 16.150 s
 *   OpenMP  1 thread ...... ______ s   speedup ____
 *   OpenMP  2 threads ..... ______ s   speedup ____
 *   OpenMP  4 threads ..... ______ s   speedup ____
 *   OpenMP  8 threads ..... ______ s   speedup ____
 *   OpenMP 16 threads ..... ______ s   speedup ____
 *   OpenMP 32 threads ..... ______ s   speedup ____
 *   CUDA .................. ______ s   speedup ____
 *
 *  Compilar: gcc -O2 -std=c11 -Wall -o kmeans_seq kmeans_seq.c -lm
 *  Executar: ./kmeans_seq data/covtype.bin 32 300
 * =====================================================================
 */