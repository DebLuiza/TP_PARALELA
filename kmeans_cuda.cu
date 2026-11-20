/*
 * =====================================================================
 *  K-MEANS - VERSAO CUDA
 *  Trabalho Pratico - Computacao Paralela - PUC Minas
 *  Integrantes: Alessandra Faria / Debora Luiza / Douglas Nicolas
 * =====================================================================
 *
 *  Base de dados: Covertype (UCI) - 581.012 pontos x 54 atributos
 *  Parametros de referencia: K = 32, max_iter = 300
 *  Resultado de referencia: 34 iteracoes, SSE aproximadamente 1.440870e+07
 *  Maquina de teste: ______________________
 *  GPU: NVIDIA GeForce RTX 3050 (sm_86)
 *
 *  TEMPOS DE EXECUCAO (apenas o laco do K-means, sem leitura do arquivo)
 *  -------------------------------------------------------------------
 *   Sequencial ............ 25.939 s
 *   OpenMP  1 thread ...... ______ s   speedup ____
 *   OpenMP  2 threads ..... ______ s   speedup ____
 *   OpenMP  4 threads ..... ______ s   speedup ____
 *   OpenMP  8 threads ..... ______ s   speedup ____
 *   OpenMP 16 threads ..... ______ s   speedup ____
 *   OpenMP 32 threads ..... ______ s   speedup ____
 *   CUDA .................. 2.685 s   speedup 9.66
 *
 *  Compilar: nvcc -O2 -arch=sm_86 -o kmeans_cuda kmeans_cuda.cu
 *  Executar: ./kmeans_cuda data/covtype.bin 32 300
 * =====================================================================
 */

#define _POSIX_C_SOURCE 199309L

#include <cuda_runtime.h>

#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define THREADS_POR_BLOCO 256

/*
 * Toda chamada da API CUDA passa por esta verificacao. Um erro de alocacao,
 * de copia ou de configuracao de kernel deve interromper o programa com a
 * operacao que falhou identificada, em vez de produzir um resultado invalido.
 */
static void verificar_cuda(cudaError_t erro, const char *operacao,
                           const char *arquivo, int linha) {
    if (erro != cudaSuccess) {
        fprintf(stderr, "Erro CUDA em %s (%s:%d): %s\n", operacao,
                arquivo, linha, cudaGetErrorString(erro));
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operacao) \
    verificar_cuda((operacao), #operacao, __FILE__, __LINE__)

static void verificar_kernel(const char *nome) {
    cudaError_t erro = cudaGetLastError();
    if (erro != cudaSuccess) {
        fprintf(stderr, "Erro ao lancar %s: %s\n", nome,
                cudaGetErrorString(erro));
        exit(EXIT_FAILURE);
    }
}

static double tempo_atual(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/*
 * O formato do dataset e compartilhado com as versoes sequencial e OpenMP:
 * dois int32 (N e D), seguidos por N * D valores float32 em ordem por linha.
 * A leitura acontece no host e fica fora da medicao do laco do K-means.
 */
static float *carregar_dataset(const char *caminho, int *n_out, int *d_out) {
    FILE *arquivo = fopen(caminho, "rb");
    if (!arquivo) {
        perror("Erro ao abrir dataset");
        exit(EXIT_FAILURE);
    }

    int cabecalho[2];
    if (fread(cabecalho, sizeof(int), 2, arquivo) != 2) {
        fprintf(stderr, "Cabecalho invalido em %s\n", caminho);
        fclose(arquivo);
        exit(EXIT_FAILURE);
    }

    int n = cabecalho[0];
    int d = cabecalho[1];
    if (n <= 0 || d <= 0) {
        fprintf(stderr, "Dimensoes invalidas: N=%d D=%d\n", n, d);
        fclose(arquivo);
        exit(EXIT_FAILURE);
    }

    size_t total = (size_t)n * (size_t)d;
    if (total > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "Dataset grande demais para a memoria do host\n");
        fclose(arquivo);
        exit(EXIT_FAILURE);
    }

    float *dados = (float *)malloc(total * sizeof(float));
    if (!dados) {
        fprintf(stderr, "Memoria insuficiente para o dataset\n");
        fclose(arquivo);
        exit(EXIT_FAILURE);
    }

    if (fread(dados, sizeof(float), total, arquivo) != total) {
        fprintf(stderr, "Arquivo truncado: esperava %zu floats\n", total);
        free(dados);
        fclose(arquivo);
        exit(EXIT_FAILURE);
    }

    fclose(arquivo);
    *n_out = n;
    *d_out = d;
    return dados;
}

/* A inicializacao deterministica permite comparar as tres versoes. */
static void inicializar_centroides(const float *dados, float *centroides,
                                   int n, int d, int k) {
    int passo = n / k;
    for (int c = 0; c < k; c++) {
        memcpy(&centroides[(size_t)c * d],
               &dados[(size_t)c * passo * d],
               (size_t)d * sizeof(float));
    }
}

static inline float distancia_quadrada_host(const float *a, const float *b,
                                            int d) {
    float soma = 0.0f;
    for (int j = 0; j < d; j++) {
        float diferenca = a[j] - b[j];
        soma += diferenca * diferenca;
    }
    return soma;
}

/*
 * O SSE e calculado no host depois do laco principal, assim como na versao
 * sequencial. Portanto, ele nao interfere na medicao do tempo do K-means.
 */
static double calcular_sse(const float *dados, const float *centroides,
                           const int *rotulos, int n, int d) {
    double sse = 0.0;
    for (int i = 0; i < n; i++) {
        const float *ponto = &dados[(size_t)i * d];
        const float *centroide = &centroides[(size_t)rotulos[i] * d];
        sse += distancia_quadrada_host(ponto, centroide, d);
    }
    return sse;
}

/* Distancia usada no device: uma thread calcula um ponto inteiro. */
__device__ __forceinline__ float distancia_quadrada_device(const float *a,
                                                           const float *b,
                                                           int d) {
    float soma = 0.0f;
    for (int j = 0; j < d; j++) {
        float diferenca = a[j] - b[j];
        soma += diferenca * diferenca;
    }
    return soma;
}

/*
 * Paralelizacao 1: uma thread por ponto.
 *
 * Cada thread le um ponto e todos os centroides, calcula o centroide mais
 * proximo e escreve somente em rotulos[i]. A unica escrita compartilhada e o
 * contador de mudancas; atomicAdd evita a condicao de corrida nesse contador.
 */
__global__ void atribuir_pontos_kernel(const float *dados,
                                       const float *centroides,
                                       int *rotulos, int *mudancas,
                                       int n, int d, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float *ponto = &dados[(size_t)i * d];
    int melhor = 0;
    float menor_distancia = FLT_MAX;

    for (int c = 0; c < k; c++) {
        const float *centroide = &centroides[(size_t)c * d];
        float distancia = distancia_quadrada_device(ponto, centroide, d);
        if (distancia < menor_distancia) {
            menor_distancia = distancia;
            melhor = c;
        }
    }

    int antigo = rotulos[i];
    rotulos[i] = melhor;
    if (antigo != melhor) atomicAdd(mudancas, 1);
}

/*
 * Paralelizacao 2, primeira etapa: cada bloco processa seus pontos e acumula
 * somas e contagens em memoria compartilhada. Isso reduz a disputa em memoria
 * global: os atomicAdd mais concorridos ficam dentro do bloco, e cada bloco
 * escreve somente uma parcial global por atributo e cluster.
 *
 * Para K=32 e D=54, o espaco compartilhado usado e:
 *   K*D*sizeof(double) + K*sizeof(int) = 13.952 bytes.
 */
__global__ void acumular_parciais_kernel(const float *dados,
                                         const int *rotulos,
                                         double *somas_parciais,
                                         int *contagens_parciais,
                                         int n, int d, int k) {
    extern __shared__ unsigned char memoria_compartilhada[];
    double *somas = (double *)memoria_compartilhada;
    int *contagens = (int *)(somas + (size_t)k * d);

    int total_atributos = k * d;

    for (int indice = threadIdx.x; indice < total_atributos;
         indice += blockDim.x) {
        somas[indice] = 0.0;
    }
    for (int c = threadIdx.x; c < k; c += blockDim.x) {
        contagens[c] = 0;
    }
    __syncthreads();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += gridDim.x * blockDim.x) {
        int c = rotulos[i];
        if (c < 0 || c >= k) continue;

        atomicAdd(&contagens[c], 1);

        const float *ponto = &dados[(size_t)i * d];
        for (int j = 0; j < d; j++) {
            atomicAdd(&somas[(size_t)c * d + j], (double)ponto[j]);
        }
    }
    __syncthreads();

    double *parciais_somas = somas_parciais +
                             (size_t)blockIdx.x * total_atributos;
    for (int indice = threadIdx.x; indice < total_atributos;
         indice += blockDim.x) {
        parciais_somas[indice] = somas[indice];
    }

    int *parciais_contagens = contagens_parciais +
                              (size_t)blockIdx.x * k;
    for (int c = threadIdx.x; c < k; c += blockDim.x) {
        parciais_contagens[c] = contagens[c];
    }
}

/*
 * Paralelizacao 2, segunda etapa: uma thread por cluster soma as parciais dos
 * blocos e atualiza todos os atributos daquele centroide. O numero de clusters
 * e pequeno, mas essa etapa ocorre na GPU e evita copiar acumuladores a cada
 * iteracao para o host.
 */
__global__ void reduzir_centroides_kernel(const double *somas_parciais,
                                          const int *contagens_parciais,
                                          float *centroides, int *contagens,
                                          int numero_blocos, int d, int k) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k) return;

    int quantidade = 0;
    for (int bloco = 0; bloco < numero_blocos; bloco++) {
        quantidade += contagens_parciais[(size_t)bloco * k + c];
    }
    contagens[c] = quantidade;

    /* Cluster vazio: o centroide permanece como estava. */
    if (quantidade == 0) return;

    for (int j = 0; j < d; j++) {
        double soma = 0.0;
        size_t indice = (size_t)c * d + j;
        for (int bloco = 0; bloco < numero_blocos; bloco++) {
            soma += somas_parciais[(size_t)bloco * k * d + indice];
        }
        centroides[indice] = (float)(soma / quantidade);
    }
}

static void salvar_rotulos(const char *caminho, const int *rotulos, int n) {
    FILE *arquivo = fopen(caminho, "w");
    if (!arquivo) {
        perror("Erro ao criar arquivo de rotulos");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; i++) {
        fprintf(arquivo, "%d\n", rotulos[i]);
    }
    fclose(arquivo);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
                "Uso: %s <dataset.bin> <K> <max_iter> [rotulos_saida.txt]\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    const char *caminho = argv[1];
    int k = atoi(argv[2]);
    int max_iter = atoi(argv[3]);
    const char *caminho_rotulos = (argc >= 5) ? argv[4] : NULL;

    int numero_dispositivos = 0;
    cudaError_t erro_dispositivo = cudaGetDeviceCount(&numero_dispositivos);
    if (erro_dispositivo != cudaSuccess || numero_dispositivos == 0) {
        fprintf(stderr, "Nenhuma GPU CUDA disponivel: %s\n",
                cudaGetErrorString(erro_dispositivo));
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp propriedades;
    CUDA_CHECK(cudaGetDeviceProperties(&propriedades, 0));

    int n, d;
    double inicio_leitura = tempo_atual();
    float *dados = carregar_dataset(caminho, &n, &d);
    double tempo_leitura = tempo_atual() - inicio_leitura;

    if (k <= 0 || k > n || max_iter <= 0) {
        fprintf(stderr, "Parametros invalidos: K=%d max_iter=%d (N=%d)\n",
                k, max_iter, n);
        free(dados);
        return EXIT_FAILURE;
    }

    size_t total_dados = (size_t)n * d;
    size_t total_centroides = (size_t)k * d;
    int numero_blocos = (n + THREADS_POR_BLOCO - 1) / THREADS_POR_BLOCO;
    size_t memoria_compartilhada = total_centroides * sizeof(double) +
                                   (size_t)k * sizeof(int);

    if (memoria_compartilhada > 48 * 1024) {
        fprintf(stderr,
                "K e D exigem %zu bytes de memoria compartilhada; "
                "o limite configurado e 49152 bytes\n",
                memoria_compartilhada);
        free(dados);
        return EXIT_FAILURE;
    }

    float *centroides = (float *)malloc(total_centroides * sizeof(float));
    int *rotulos = (int *)malloc((size_t)n * sizeof(int));
    long *contagens_host = (long *)calloc((size_t)k, sizeof(long));
    if (!centroides || !rotulos || !contagens_host) {
        fprintf(stderr, "Memoria insuficiente no host\n");
        free(dados);
        free(centroides);
        free(rotulos);
        free(contagens_host);
        return EXIT_FAILURE;
    }

    inicializar_centroides(dados, centroides, n, d, k);

    float *dados_device = NULL;
    float *centroides_device = NULL;
    int *rotulos_device = NULL;
    int *mudancas_device = NULL;
    int *contagens_device = NULL;
    double *somas_parciais_device = NULL;
    int *contagens_parciais_device = NULL;

    CUDA_CHECK(cudaMalloc((void **)&dados_device,
                          total_dados * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&centroides_device,
                          total_centroides * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&rotulos_device,
                          (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&mudancas_device, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&contagens_device,
                          (size_t)k * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&somas_parciais_device,
                          (size_t)numero_blocos * total_centroides *
                              sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&contagens_parciais_device,
                          (size_t)numero_blocos * k * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dados_device, dados, total_dados * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(centroides_device, centroides,
                          total_centroides * sizeof(float),
                          cudaMemcpyHostToDevice));
    /* -1 faz com que a primeira atribuicao conte todos os pontos como mudanca. */
    CUDA_CHECK(cudaMemset(rotulos_device, 0xFF, (size_t)n * sizeof(int)));

    dim3 bloco(THREADS_POR_BLOCO);
    dim3 grade_atribuicao((n + THREADS_POR_BLOCO - 1) /
                          THREADS_POR_BLOCO);
    dim3 grade_reducao((k + THREADS_POR_BLOCO - 1) /
                       THREADS_POR_BLOCO);

    cudaEvent_t evento_inicio, evento_fim;
    CUDA_CHECK(cudaEventCreate(&evento_inicio));
    CUDA_CHECK(cudaEventCreate(&evento_fim));

    int iteracoes = 0;
    int mudancas = 0;

    /* A transferencia inicial fica fora do tempo, assim como a leitura do arquivo. */
    CUDA_CHECK(cudaEventRecord(evento_inicio));
    while (iteracoes < max_iter) {
        CUDA_CHECK(cudaMemset(mudancas_device, 0, sizeof(int)));

        atribuir_pontos_kernel<<<grade_atribuicao, bloco>>>(
            dados_device, centroides_device, rotulos_device,
            mudancas_device, n, d, k);
        verificar_kernel("atribuir_pontos_kernel");

        /* A copia de um unico inteiro sincroniza a atribuicao antes da decisao. */
        CUDA_CHECK(cudaMemcpy(&mudancas, mudancas_device, sizeof(int),
                              cudaMemcpyDeviceToHost));
        iteracoes++;

        if (mudancas == 0) break;

        acumular_parciais_kernel<<<numero_blocos, bloco, memoria_compartilhada>>>(
            dados_device, rotulos_device, somas_parciais_device,
            contagens_parciais_device, n, d, k);
        verificar_kernel("acumular_parciais_kernel");

        reduzir_centroides_kernel<<<grade_reducao, bloco>>>(
            somas_parciais_device, contagens_parciais_device,
            centroides_device, contagens_device, numero_blocos, d, k);
        verificar_kernel("reduzir_centroides_kernel");
    }
    CUDA_CHECK(cudaEventRecord(evento_fim));
    CUDA_CHECK(cudaEventSynchronize(evento_fim));

    float tempo_kmeans_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&tempo_kmeans_ms,
                                   evento_inicio, evento_fim));
    double tempo_kmeans = (double)tempo_kmeans_ms / 1000.0;

    CUDA_CHECK(cudaMemcpy(rotulos, rotulos_device, (size_t)n * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(centroides, centroides_device,
                          total_centroides * sizeof(float),
                          cudaMemcpyDeviceToHost));

    double sse = calcular_sse(dados, centroides, rotulos, n, d);
    for (int i = 0; i < n; i++) {
        if (rotulos[i] >= 0 && rotulos[i] < k) {
            contagens_host[rotulos[i]]++;
        }
    }

    printf("Versao       : CUDA\n");
    printf("GPU          : %s\n", propriedades.name);
    printf("Dataset      : %s (N=%d, D=%d)\n", caminho, n, d);
    printf("K            : %d\n", k);
    printf("Iteracoes    : %d%s\n", iteracoes,
           mudancas == 0 ? " (convergiu)" : " (atingiu max_iter)");
    printf("SSE          : %.6e\n", sse);
    printf("Tempo leitura: %.3f s\n", tempo_leitura);
    printf("Tempo K-means: %.3f s\n", tempo_kmeans);
    printf("Tam. clusters:");
    for (int c = 0; c < k; c++) {
        printf(" %ld", contagens_host[c]);
    }
    printf("\n");

    if (caminho_rotulos) {
        salvar_rotulos(caminho_rotulos, rotulos, n);
    }

    CUDA_CHECK(cudaEventDestroy(evento_inicio));
    CUDA_CHECK(cudaEventDestroy(evento_fim));
    CUDA_CHECK(cudaFree(dados_device));
    CUDA_CHECK(cudaFree(centroides_device));
    CUDA_CHECK(cudaFree(rotulos_device));
    CUDA_CHECK(cudaFree(mudancas_device));
    CUDA_CHECK(cudaFree(contagens_device));
    CUDA_CHECK(cudaFree(somas_parciais_device));
    CUDA_CHECK(cudaFree(contagens_parciais_device));

    free(dados);
    free(centroides);
    free(rotulos);
    free(contagens_host);
    return EXIT_SUCCESS;
}
