/*
 * =====================================================================
 *  K-MEANS - VERSAO OPENMP (CPU MULTICORE)
 *  Trabalho Prático - Computação Paralela - PUC Minas
 *  Integrantes: Alessandra Faria / Débora Luiza / Douglas Nícolas
 * =====================================================================
 *
 *  Base de dados: Covertype (UCI) - 581.012 pontos x 54 atributos
 *  Parâmetros: K = 32, max_iter = 300
 *  Resultado de referência: 34 iterações, SSE = 1.440870e+07
 *  Máquina de teste: AMD Ryzen 7 5700X3D (8 núcleos/16 threads lógicas),
 *                    NVIDIA GeForce RTX 4060 Ti, Windows 11 Pro
 *  Método: mediana de 5 execuções após 1 aquecimento por configuração
 *
 *  TEMPOS DE EXECUCAO (apenas o laco do K-means, sem leitura do arquivo):
 *   Sequencial ............ 16.180 s   speedup 1.00
 *   OpenMP  1 thread ...... 16.974 s   speedup 0.95
 *   OpenMP  2 threads .....  8.619 s   speedup 1.88
 *   OpenMP  4 threads .....  4.406 s   speedup 3.67
 *   OpenMP  8 threads .....  2.536 s   speedup 6.38
 *   OpenMP 16 threads .....  1.744 s   speedup 9.28
 *   OpenMP 32 threads .....  1.738 s   speedup 9.31 (sobrescricao)
 *   CUDA ..................  0.614 s   speedup 26.35
 * 
 *  Resultado: 34 iteracoes, SSE = 1.440870e+07
 *
 * OBS: Os testes foram realizados na mesma maquina. Como a CPU possui
 * 16 threads logicas, o teste com 32 threads usa sobrescricao.
 *
 *  Compilar: gcc -O2 -std=c11 -Wall -Wextra -fopenmp -o kmeans_omp.exe kmeans_omp.c -lm
 *  Executar (PowerShell): $env:OMP_NUM_THREADS=4
 *                         .\kmeans_omp.exe .\data\covtype.bin 32 300
 */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <time.h>
#include <omp.h>

//devolve o tempo em segundos com precisão de nanossegundos
static double tempo_atual(void) {
    struct timespec ts;
    //mede o tempo real decorrido, o clock() mede o tempo somado de CPU das threads
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static float *carregar_dataset(const char *caminho, int *n_out, int *d_out) {
    FILE *f = fopen(caminho, "rb");
    if (!f) { perror("Erro ao abrir dataset"); exit(EXIT_FAILURE); }
    // o arquivo possui o formato de 2 inteiros (N pontos, D atributos) seguidos de N×D floats

    int cabecalho[2];
    //ler com um único fread é bem mais rápido que interpretar um CSV.
    if (fread(cabecalho, sizeof(int), 2, f) != 2) {
        fprintf(stderr, "Cabecalho invalido em %s\n", caminho);
        exit(EXIT_FAILURE);
    }
    int n = cabecalho[0], d = cabecalho[1];
    if (n <= 0 || d <= 0) {
        fprintf(stderr, "Dimensoes invalidas: N=%d D=%d\n", n, d);
        exit(EXIT_FAILURE);
    }

    // N×D pode passar do limite de um int. 
    // O cast para size_t evita overflow com bases grandes
    size_t total = (size_t)n * (size_t)d;
    float *X = malloc(total * sizeof(float));
    if (!X) { fprintf(stderr, "Memoria insuficiente\n"); exit(EXIT_FAILURE); }

    if (fread(X, sizeof(float), total, f) != total) {
        fprintf(stderr, "Arquivo truncado: esperava %zu floats\n", total);
        exit(EXIT_FAILURE);
    }
    fclose(f);
    *n_out = n;
    *d_out = d;
    return X;
}

//Calcula a distância euclidiana sem a raiz quadrada
//obs: não precisamos da raiz porque só queremos saber qual centróide é o mais próximo
static inline float distancia_quadrada(const float *a, const float *b, int d) {
    //essa função roda N × K vezes por iteração e sqrt é uma operação cara.
    //se x < y, então √x < √y

    float s = 0.0f;
    for (int j = 0; j < d; j++) {
        float diff = a[j] - b[j];
        s += diff * diff;
    }
    return s;
}


//O K-means precisa de K centróides iniciais.
// Escolhemos pontos espalhados pela base: o centróide c recebe uma cópia do ponto c × (N/K).
static void inicializar_centroides(const float *X, float *C, int n, int d, int k) {
    //Não escolhemos sortidos pois com sorteio cada execução daria um resultado diferente. 
    // Sendo determinístico, as versões sequencial, OpenMP e CUDA vão partit dos mesmos centróides
    long passo = n / k;
    for (int c = 0; c < k; c++) {
        //O memcpy copia os d floats do ponto de uma vez
        memcpy(&C[(size_t)c * d], &X[(size_t)c * passo * d], d * sizeof(float));
    }
}


/*
 * PARALELIZACAO 1: atribuicao dos pontos.
 * Cada ponto e independente: as threads somente leem X e C e cada uma
 * modifica uma posicao diferente de labels[i].
 * A variavel mudancas e compartilhada logicamente; reduction cria uma
 * contagem privada por thread e soma tudo ao final, sem condicao de corrida.
 * schedule(static) distribui os pontos em blocos fixos entre as threads.
 */
static long atribuir_pontos(const float *X, const float *C, int *labels,
                            int n, int d, int k) {
    long mudancas = 0;
    int i;

    #pragma omp parallel for reduction(+:mudancas) schedule(static)
    for (i = 0; i < n; i++) {
        const float *p = &X[(size_t)i * d];
        int melhor = 0;
        float melhor_dist = FLT_MAX;

        for (int c = 0; c < k; c++) {
            float dd = distancia_quadrada(p, &C[(size_t)c * d], d);
            if (dd < melhor_dist) {
                melhor_dist = dd;
                melhor = c;
            }
        }

        if (labels[i] != melhor) {
            labels[i] = melhor;
            mudancas++;
        }
    }
    return mudancas;
}

/*
 * PARALELIZACAO 2: recalcular a media dos centroides.
 * Nao e seguro usar um simples "parallel for" somando em somas[c] e
 * contagens[c]: duas threads poderiam escrever ao mesmo tempo no mesmo local.
 *
 * SOLUCAO: cada thread tem sua propria linha de somas_locais e
 * contagens_locais, identificada por omp_get_thread_num().
 * Depois de terminar a regiao paralela, juntamos os acumuladores e
 * calculamos a media. Nao precisamos de atomic nem critical dentro do
 * loop de pontos (o que causaria muita espera entre threads).
 *
 * Os buffers locais sao criados UMA vez no main, nao a cada iteracao.
 */
static void atualizar_centroides(const float *X, float *C, const int *labels,
                                 double *somas, long *contagens,
                                 double *somas_locais, long *contagens_locais,
                                 int n, int d, int k, int max_threads) {
    size_t kd = (size_t)k * d;

    // Zera os acumuladores individuais de todas as threads.
    memset(somas_locais, 0, (size_t)max_threads * kd * sizeof(double));
    memset(contagens_locais, 0, (size_t)max_threads * k * sizeof(long));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num(); // Numero da thread atual: 0, 1, 2...
        double *minhas_somas = &somas_locais[(size_t)tid * kd];
        long *minhas_contagens = &contagens_locais[(size_t)tid * k];
        int i;

        // A diretiva for divide as iteracoes do laco entre as threads.
        #pragma omp for schedule(static)
        for (i = 0; i < n; i++) {
            int c = labels[i];
            const float *p = &X[(size_t)i * d];
            double *s = &minhas_somas[(size_t)c * d];

            for (int j = 0; j < d; j++) {
                s[j] += p[j];
            }
            minhas_contagens[c]++;
        }
    } // Aqui todas as threads ja terminaram suas somas locais.

    // Junta as somas locais. Esta etapa pequena continua sequencial.
    memset(somas, 0, kd * sizeof(double));
    memset(contagens, 0, (size_t)k * sizeof(long));

    for (int t = 0; t < max_threads; t++) {
        for (int c = 0; c < k; c++) {
            contagens[c] += contagens_locais[(size_t)t * k + c];
            for (int j = 0; j < d; j++) {
                somas[(size_t)c * d + j] +=
                    somas_locais[(size_t)t * kd + (size_t)c * d + j];
            }
        }
    }

    // O centroide e a media dos pontos do seu cluster.
    for (int c = 0; c < k; c++) {
        if (contagens[c] == 0) continue; // Evita divisao por zero.
        for (int j = 0; j < d; j++) {
            C[(size_t)c * d + j] =
                (float)(somas[(size_t)c * d + j] / contagens[c]);
        }
    }
}

//O SSE (Sum of Squared Errors) é a soma das distâncias ao quadrado de cada ponto até o centróide do seu cluster. 
// Quanto menor, mais “compactos” são os clusters.
static double calcular_sse(const float *X, const float *C, const int *labels,
                           int n, int d) {

    // OpenMP e CUDA têm que dar o mesmo SSE (com diferença só nas últimas casas).
    double sse = 0.0;
    for (int i = 0; i < n; i++) {
        sse += distancia_quadrada(&X[(size_t)i * d], &C[(size_t)labels[i] * d], d);
    }
    return sse;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Uso: %s <dataset.bin> <K> <max_iter> [rotulos_saida.txt]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char *caminho = argv[1];
    int k = atoi(argv[2]);
    int max_iter = atoi(argv[3]);
    const char *caminho_rotulos = (argc >= 5) ? argv[4] : NULL;

    int n, d;
    double t_leitura = tempo_atual();
    float *X = carregar_dataset(caminho, &n, &d);
    t_leitura = tempo_atual() - t_leitura;

    if (k <= 0 || k > n || max_iter <= 0) {
        fprintf(stderr, "Parametros invalidos: K=%d max_iter=%d (N=%d)\n", k, max_iter, n);
        return EXIT_FAILURE;
    }

    int max_threads = omp_get_max_threads();

    float  *C         = malloc((size_t)k * d * sizeof(float));
    int    *labels    = malloc((size_t)n * sizeof(int));
    double *somas     = malloc((size_t)k * d * sizeof(double));
    long   *contagens = malloc((size_t)k * sizeof(long));
    // Buffers privados para a etapa de atualizacao dos centroides.
    double *somas_locais = calloc((size_t)max_threads * k * d, sizeof(double));
    long *contagens_locais = calloc((size_t)max_threads * k, sizeof(long));
    if (!C || !labels || !somas || !contagens ||
        !somas_locais || !contagens_locais) {
        fprintf(stderr, "Memoria insuficiente\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < n; i++) labels[i] = -1;
    inicializar_centroides(X, C, n, d, k);

    double t0 = tempo_atual();
    int iter = 0;
    long mudancas = 0;
    while (iter < max_iter) {
        mudancas = atribuir_pontos(X, C, labels, n, d, k);
        iter++;
        if (mudancas == 0) break;
        atualizar_centroides(X, C, labels, somas, contagens,
                             somas_locais, contagens_locais,
                             n, d, k, max_threads);
    }
    double t_kmeans = tempo_atual() - t0;

    double sse = calcular_sse(X, C, labels, n, d);

    printf("Versao       : OpenMP\n");
    printf("Threads      : %d\n", max_threads);
    printf("Dataset      : %s (N=%d, D=%d)\n", caminho, n, d);
    printf("K            : %d\n", k);
    printf("Iteracoes    : %d%s\n", iter, mudancas == 0 ? " (convergiu)" : " (atingiu max_iter)");
    printf("SSE          : %.6e\n", sse);
    printf("Tempo leitura: %.3f s\n", t_leitura);
    printf("Tempo K-means: %.3f s\n", t_kmeans);

    for (int c = 0; c < k; c++) contagens[c] = 0;
    for (int i = 0; i < n; i++) contagens[labels[i]]++;
    printf("Tam. clusters:");
    for (int c = 0; c < k; c++) printf(" %ld", contagens[c]);
    printf("\n");

    if (caminho_rotulos) {
        FILE *saida = fopen(caminho_rotulos, "w");
        if (!saida) { perror("Erro ao criar arquivo de rotulos"); return EXIT_FAILURE; }
        for (int i = 0; i < n; i++) fprintf(saida, "%d\n", labels[i]);
        fclose(saida);
    }

    free(X); free(C); free(labels); free(somas); free(contagens);
    free(somas_locais); free(contagens_locais);
    return EXIT_SUCCESS;
}
