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
 *  Maquina de teste: AMD Ryzen 7 5700X3D (8 nucleos/16 threads logicas),
 *                    NVIDIA GeForce RTX 4060 Ti, Windows 11 Pro
 *  GPU/arquitetura: NVIDIA GeForce RTX 4060 Ti (sm_89)
 *  Metodo: mediana de 5 execucoes apos 1 aquecimento por configuracao
 *
 *  TEMPOS DE EXECUCAO (apenas o laco do K-means, sem leitura do arquivo)
 *  -------------------------------------------------------------------
 *   Sequencial ............ 16.180 s   speedup 1.00
 *   OpenMP  1 thread ...... 16.974 s   speedup 0.95
 *   OpenMP  2 threads .....  8.619 s   speedup 1.88
 *   OpenMP  4 threads .....  4.406 s   speedup 3.67
 *   OpenMP  8 threads .....  2.536 s   speedup 6.38
 *   OpenMP 16 threads .....  1.744 s   speedup 9.28
 *   OpenMP 32 threads .....  1.738 s   speedup 9.31 (sobrescricao)
 *   CUDA ..................  0.614 s   speedup 26.35
 *
 *  Compilar (Linux): nvcc -O2 -arch=sm_89 -o kmeans_cuda kmeans_cuda.cu
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

/* CONFIGURACAO: define o tamanho dos blocos.
 * Cada bloco CUDA usa 256 threads.
 * O valor organiza o trabalho entre threads e blocos, sem mudar a formula.
 * Ele e usado pela atribuicao, acumulacao e reducao.
 *
 */

/* ERRO: verifica chamadas da API CUDA.
 * Erros de alocacao, copia ou configuracao interrompem o programa.
 * A operacao que falhou e exibida para facilitar a localizacao do problema.
 * cudaGetDeviceCount tem tratamento especial no main.
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
    /* ERRO: verifica o lancamento de um kernel.
     * Detecta imediatamente erros de configuracao, como uma grade invalida.
     * Erros durante a execucao aparecem em operacoes posteriores que sincronizam.
     * Exemplos: cudaMemcpy e cudaEventSynchronize.
     *
     */
    cudaError_t erro = cudaGetLastError();
    if (erro != cudaSuccess) {
        fprintf(stderr, "Erro ao lancar %s: %s\n", nome,
                cudaGetErrorString(erro));
        exit(EXIT_FAILURE);
    }
}

static double tempo_atual(void) {
    /* TEMPO: mede o relogio do host.
     * Serve para medir a leitura; o laco CUDA usa cudaEvent na GPU.
     * Isso evita confundir o tempo do host com a execucao assincrona do device.
     *
     */
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* DADOS: define o formato e o local da leitura.
 * O arquivo tem dois int32 (N e D), seguidos por N*D float32 em row-major.
 * A leitura ocorre no host e fica fora da medicao do laco do K-means.
 *
 */
static float *carregar_dataset(const char *caminho, int *n_out, int *d_out) {
    /* DADOS: carrega o dataset para a memoria do host.
     * O ponto i ocupa as posicoes i*d ate i*d+d-1 no vetor contiguo.
     * Essa organizacao row-major e mantida quando os dados vao para a GPU.
     *
     */
    FILE *arquivo = fopen(caminho, "rb");
    if (!arquivo) {
        perror("Erro ao abrir dataset");
        exit(EXIT_FAILURE);
    }

    /* DADOS: le N pontos e D atributos do cabecalho. */
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

    /* MEMORIA: calcula o tamanho da alocacao com size_t.
     * O tipo evita limitar N*D ao intervalo de um int.
     * A verificacao seguinte evita overflow ao multiplicar por sizeof(float).
     *
     */
    size_t total = (size_t)n * (size_t)d;
    if (total > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "Dataset grande demais para a memoria do host\n");
        fclose(arquivo);
        exit(EXIT_FAILURE);
    }

    /* MEMORIA: reserva o vetor que permanece no host ate a copia inicial. */
    float *dados = (float *)malloc(total * sizeof(float));
    if (!dados) {
        fprintf(stderr, "Memoria insuficiente para o dataset\n");
        fclose(arquivo);
        exit(EXIT_FAILURE);
    }

    /* DADOS: le todos os floats de uma vez, sem interpretar CSV. */
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

/* INICIALIZACAO: escolhe centroides deterministicos para comparar versoes. */
static void inicializar_centroides(const float *dados, float *centroides,
                                   int n, int d, int k) {
    /* INICIALIZACAO: copia o ponto c*(N/K) para o centroide c.
     * Nao ha sorteio; as versoes partem dos mesmos centroides.
     * Isso facilita comparar rotulos, SSE e numero de iteracoes.
     *
     */
    int passo = n / k;
    for (int c = 0; c < k; c++) {
        /* MEMORIA: copia os d atributos consecutivos do ponto escolhido. */
        memcpy(&centroides[(size_t)c * d],
               &dados[(size_t)c * passo * d],
               (size_t)d * sizeof(float));
    }
}

static inline float distancia_quadrada_host(const float *a, const float *b,
                                            int d) {
    /* CALCULO: mede a distancia euclidiana ao quadrado no host.
     * A funcao soma (a[j]-b[j])^2 para os D atributos.
     * A raiz quadrada e desnecessaria porque nao muda a ordem das distancias.
     *
     *
     */
    float soma = 0.0f;
    for (int j = 0; j < d; j++) {
        float diferenca = a[j] - b[j];
        soma += diferenca * diferenca;
    }
    return soma;
}

/* SSE: calcula a qualidade final no host.
 * A rotina roda depois do laco e fica fora da medicao do K-means.
 * Ela segue a mesma ideia da versao sequencial para facilitar a comparacao.
 */
static double calcular_sse(const float *dados, const float *centroides,
                           const int *rotulos, int n, int d) {
    /* SSE: soma a distancia de cada ponto ao seu centroide.
     * O acumulador e double para reduzir o erro da soma.
     * A funcao roda uma vez, depois do treinamento.
     *
     *
     */
    double sse = 0.0;
    for (int i = 0; i < n; i++) {
        const float *ponto = &dados[(size_t)i * d];
        const float *centroide = &centroides[(size_t)rotulos[i] * d];
        sse += distancia_quadrada_host(ponto, centroide, d);
    }
    return sse;
}

/* CALCULO: mede a distancia euclidiana ao quadrado no device.
 * Uma thread percorre os D atributos e compara um ponto com um centroide.
 *
 *
 */
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

/* PARALELIZACAO 1: atribui um cluster para cada ponto.
 * Uma thread testa todos os centroides e grava somente rotulos[i].
 * O contador compartilhado usa atomicAdd para nao perder incrementos.
 *
 *
 */
__global__ void atribuir_pontos_kernel(const float *dados,
                                       const float *centroides,
                                       int *rotulos, int *mudancas,
                                       int n, int d, int k) {
    /* INDICE: transforma bloco e thread em um indice global.
     * Pode haver threads sobrando quando N nao e multiplo de 256.
     * Por isso a guarda precisa acontecer antes de acessar dados[i].
     *
     *
     */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    /* MEMORIA: esta thread acessa somente o vetor do ponto i. */
    const float *ponto = &dados[(size_t)i * d];
    int melhor = 0;
    float menor_distancia = FLT_MAX;

    /* CALCULO: escolhe o centroide de menor distancia.
     * O teste percorre K centroides e, em empate, conserva o primeiro.
     * A comparacao usa <, assim como na versao sequencial.
     *
     *
     */
    for (int c = 0; c < k; c++) {
        const float *centroide = &centroides[(size_t)c * d];
        float distancia = distancia_quadrada_device(ponto, centroide, d);
        if (distancia < menor_distancia) {
            menor_distancia = distancia;
            melhor = c;
        }
    }

    /* ROTULOS: grava o cluster escolhido para o ponto i.
     * Cada thread escreve em uma posicao diferente de rotulos.
     * O valor antigo serve para detectar mudanca nesta iteracao.
     *
     */
    int antigo = rotulos[i];
    rotulos[i] = melhor;

    /* ATOMIC: conta mudancas sem condicao de corrida.
     * Varios pontos podem incrementar o mesmo inteiro ao mesmo tempo.
     * A ordem dos incrementos nao importa; somente o total e usado.
     *
     *
     */
    if (antigo != melhor) atomicAdd(mudancas, 1);
}

/* PARALELIZACAO 2A: acumula somas parciais por bloco.
 * O bloco usa memoria compartilhada para reduzir atomics na memoria global.
 * Cada bloco grava uma parcial por cluster e atributo.
 *
 * Para K=32 e D=54, o uso e K*D*sizeof(double)+K*sizeof(int): 13.952 bytes.
 *
 *
 */
__global__ void acumular_parciais_kernel(const float *dados,
                                         const int *rotulos,
                                         double *somas_parciais,
                                         int *contagens_parciais,
                                         int n, int d, int k) {
    /* MEMORIA: separa a shared memory em somas e contagens.
     * Layout: [k*d doubles de somas][k ints de contagens].
     * O tamanho depende de K e D e vem do terceiro argumento do lancamento.
     *
     *
     *
     */
    extern __shared__ unsigned char memoria_compartilhada[];
    double *somas = (double *)memoria_compartilhada;
    int *contagens = (int *)(somas + (size_t)k * d);

    int total_atributos = k * d;

    /* INICIALIZACAO: threads zeram os acumuladores em conjunto.
     * Cada thread percorre indices separados por blockDim.x.
     *
     */
    for (int indice = threadIdx.x; indice < total_atributos;
         indice += blockDim.x) {
        somas[indice] = 0.0;
    }
    for (int c = threadIdx.x; c < k; c += blockDim.x) {
        contagens[c] = 0;
    }

    /* SINCRONIZACAO: espera o zeramento antes de acumular. */
    __syncthreads();

    /* PARALELIZACAO: distribui pontos com um laco grid-stride.
     * Uma thread pode reutilizar-se em indices separados pelo tamanho da grade.
     * Isso permite processar mais pontos que o total de threads lancadas.
     *
     *
     */
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += gridDim.x * blockDim.x) {
        int c = rotulos[i];

        /* VALIDACAO: ignora rotulos fora do intervalo [0, k). */
        if (c < 0 || c >= k) continue;

        /* ATOMIC: protege contagens e somas do mesmo cluster.
         * A disputa fica na shared memory e e local ao bloco.
         * Isso reduz a pressao sobre atomics na memoria global.
         *
         *
         */
        atomicAdd(&contagens[c], 1);

        const float *ponto = &dados[(size_t)i * d];
        for (int j = 0; j < d; j++) {
            /* PRECISAO: acumula o atributo em double. */
            atomicAdd(&somas[(size_t)c * d + j], (double)ponto[j]);
        }
    }

    /* SINCRONIZACAO: espera os atomics antes de copiar as parciais. */
    __syncthreads();

    /* MEMORIA: salva os acumuladores em regioes proprias da memoria global.
     * A shared memory desaparece quando o bloco termina.
     * As parciais ficam disponiveis para o kernel de reducao.
     *
     *
     */
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

/* PARALELIZACAO 2B: reduz parciais e atualiza os centroides.
 * Uma thread por cluster soma as parciais e calcula seus D atributos.
 * A reducao continua na GPU e evita copiar acumuladores a cada iteracao.
 *
 *
 */
__global__ void reduzir_centroides_kernel(const double *somas_parciais,
                                          const int *contagens_parciais,
                                          float *centroides, int *contagens,
                                          int numero_blocos, int d, int k) {
    /* PARALELIZACAO: cada thread passa a cuidar de um cluster.
     * Como K e pequeno, a thread percorre as parciais de todos os blocos.
     * Depois ela calcula os D atributos do seu proprio centroide.
     *
     *
     */
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k) return;

    /* REDUCAO: soma as contagens parciais do cluster c. */
    int quantidade = 0;
    for (int bloco = 0; bloco < numero_blocos; bloco++) {
        quantidade += contagens_parciais[(size_t)bloco * k + c];
    }
    contagens[c] = quantidade;

    /* CLUSTER VAZIO: preserva o centroide da iteracao anterior.
     * Sem pontos, nao existe media e a divisao seria invalida.
     *
     */
    if (quantidade == 0) return;

    /* MEDIA: soma parciais e divide pela quantidade do cluster.
     * Cada thread escreve somente no proprio centroide.
     *
     *
     */
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
    /* SAIDA: salva um rotulo por linha no host.
     * A funcao e chamada somente depois da copia dos resultados da GPU.
     * Cada linha corresponde a um ponto da base.
     */
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
    /* ARGUMENTOS: le dataset, K, limite e saida opcional.
     * A validacao acontece antes das alocacoes para falhar cedo.
     *
     *
     *
     */
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

    /* GPU: verifica disponibilidade e seleciona o dispositivo zero.
     * A escolha explicita deixa o programa deterministico em varias GPUs.
     *
     *
     */
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

    /* DADOS: carrega a base na CPU e mede sua leitura separadamente. */
    int n, d;
    double inicio_leitura = tempo_atual();
    float *dados = carregar_dataset(caminho, &n, &d);
    double tempo_leitura = tempo_atual() - inicio_leitura;

    /* VALIDACAO: garante K valido e pelo menos uma iteracao. */
    if (k <= 0 || k > n || max_iter <= 0) {
        fprintf(stderr, "Parametros invalidos: K=%d max_iter=%d (N=%d)\n",
                k, max_iter, n);
        free(dados);
        return EXIT_FAILURE;
    }

    /* MEMORIA: calcula tamanhos de dados, centroides e parciais.
     * dados tem N*D floats; centroides tem K*D floats.
     * Cada bloco recebe uma parcial de K*D e uma de K.
     *
     *
     */
    size_t total_dados = (size_t)n * d;
    size_t total_centroides = (size_t)k * d;

    /* CONFIGURACAO: calcula o numero de blocos para N pontos.
     * O arredondamento para cima cobre o ultimo bloco incompleto.
     *
     *
     *
     */
    int numero_blocos = (n + THREADS_POR_BLOCO - 1) / THREADS_POR_BLOCO;

    /* VALIDACAO: confere o limite de memoria compartilhada por bloco.
     * A verificacao evita lancar uma configuracao que a GPU nao aceite.
     *
     */
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

    /* MEMORIA: aloca buffers que permanecem no host.
     * centroides e inicializado aqui; rotulos e contagens recebem resultados.
     *
     *
     */
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

    /* MEMORIA: separa ponteiros do host e do device.
     * Nomes com sufixo _device apontam para a memoria global da GPU.
     * Os demais ponteiros referem-se a buffers da CPU.
     *
     */
    float *dados_device = NULL;
    float *centroides_device = NULL;
    int *rotulos_device = NULL;
    int *mudancas_device = NULL;
    int *contagens_device = NULL;
    double *somas_parciais_device = NULL;
    int *contagens_parciais_device = NULL;

    /* MEMORIA: aloca na GPU os buffers usados em cada iteracao. */
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

    /* TRANSFERENCIA: envia dados e centroides para a GPU uma unica vez.
     * Durante o laco, os kernels usam as versoes que permanecem no device.
     *
     */
    CUDA_CHECK(cudaMemcpy(dados_device, dados, total_dados * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(centroides_device, centroides,
                          total_centroides * sizeof(float),
                          cudaMemcpyHostToDevice));
    /* INICIALIZACAO: marca todos os rotulos como -1.
     * Assim, a primeira atribuicao conta todos os pontos como mudancas.
     * O padrao 0xFF representa -1 em int de complemento de dois.
     *
     */
    CUDA_CHECK(cudaMemset(rotulos_device, 0xFF, (size_t)n * sizeof(int)));

    /* CONFIGURACAO: define grades e blocos dos kernels.
     * grade_atribuicao usa uma thread por ponto.
     * grade_reducao usa uma thread por cluster; bloco tem 256 threads.
     *
     */
    dim3 bloco(THREADS_POR_BLOCO);
    dim3 grade_atribuicao((n + THREADS_POR_BLOCO - 1) /
                          THREADS_POR_BLOCO);
    dim3 grade_reducao((k + THREADS_POR_BLOCO - 1) /
                       THREADS_POR_BLOCO);

    /* TEMPO: cria eventos para medir a linha do tempo da GPU. */
    cudaEvent_t evento_inicio, evento_fim;
    CUDA_CHECK(cudaEventCreate(&evento_inicio));
    CUDA_CHECK(cudaEventCreate(&evento_fim));

    int iteracoes = 0;
    int mudancas = 0;

    /* TEMPO: inicia a medicao depois das transferencias iniciais.
     * O evento fica imediatamente antes do laco de treinamento.
     *
     */
    CUDA_CHECK(cudaEventRecord(evento_inicio));
    while (iteracoes < max_iter) {
        /* LOOP: repete atribuicao, convergencia e atualizacao dos centroides.
         * A ordem e a mesma do sequencial, mas cada etapa usa kernels.
         *
         *
         */
        /* INICIALIZACAO: zera o contador antes da atribuicao. */
        CUDA_CHECK(cudaMemset(mudancas_device, 0, sizeof(int)));

        /* PARALELIZACAO 1: executa atribuicao, uma thread por ponto. */
        atribuir_pontos_kernel<<<grade_atribuicao, bloco>>>(
            dados_device, centroides_device, rotulos_device,
            mudancas_device, n, d, k);
        verificar_kernel("atribuir_pontos_kernel");

        /* SINCRONIZACAO: copia mudancas e espera a atribuicao terminar.
         * O host precisa do contador completo antes de decidir se para.
         *
         *
         */
        CUDA_CHECK(cudaMemcpy(&mudancas, mudancas_device, sizeof(int),
                              cudaMemcpyDeviceToHost));
        iteracoes++;

        if (mudancas == 0) break;

        /* PARALELIZACAO 2A: acumula somas parciais por bloco. */
        acumular_parciais_kernel<<<numero_blocos, bloco, memoria_compartilhada>>>(
            dados_device, rotulos_device, somas_parciais_device,
            contagens_parciais_device, n, d, k);
        verificar_kernel("acumular_parciais_kernel");

        /* PARALELIZACAO 2B: reduz parciais, uma thread por cluster. */
        reduzir_centroides_kernel<<<grade_reducao, bloco>>>(
            somas_parciais_device, contagens_parciais_device,
            centroides_device, contagens_device, numero_blocos, d, k);
        verificar_kernel("reduzir_centroides_kernel");
    }
    /* TEMPO: encerra a medicao depois do ultimo kernel.
     * A sincronizacao garante que todos os kernels terminaram.
     *
     */
    CUDA_CHECK(cudaEventRecord(evento_fim));
    CUDA_CHECK(cudaEventSynchronize(evento_fim));

    float tempo_kmeans_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&tempo_kmeans_ms,
                                   evento_inicio, evento_fim));
    double tempo_kmeans = (double)tempo_kmeans_ms / 1000.0;

    /* TRANSFERENCIA: traz rotulos e centroides finais para o host.
     * Esses dados serao usados no SSE, na validacao e na saida opcional.
     *
     */
    CUDA_CHECK(cudaMemcpy(rotulos, rotulos_device, (size_t)n * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(centroides, centroides_device,
                          total_centroides * sizeof(float),
                          cudaMemcpyDeviceToHost));

    /* SSE: calcula a qualidade final usando a rotina do host. */
    double sse = calcular_sse(dados, centroides, rotulos, n, d);

    /* VALIDACAO: reconta no host o tamanho de cada cluster. */
    for (int i = 0; i < n; i++) {
        if (rotulos[i] >= 0 && rotulos[i] < k) {
            contagens_host[rotulos[i]]++;
        }
    }

    /* SAIDA: imprime valores para comparar com a versao sequencial. */
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

    /* SAIDA: salva rotulos somente se o caminho opcional foi informado. */
    if (caminho_rotulos) {
        salvar_rotulos(caminho_rotulos, rotulos, n);
    }

    /* LIMPEZA: libera eventos e buffers da GPU e da CPU.
     * A liberacao explicita evita vazamentos de memoria ao encerrar.
     *
     *
     */
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
