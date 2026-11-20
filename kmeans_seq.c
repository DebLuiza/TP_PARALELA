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
 *   Sequencial ............ 25.939 s
 *   OpenMP  1 thread ...... ______ s   speedup ____
 *   OpenMP  2 threads ..... ______ s   speedup ____
 *   OpenMP  4 threads ..... ______ s   speedup ____
 *   OpenMP  8 threads ..... ______ s   speedup ____
 *   OpenMP 16 threads ..... ______ s   speedup ____
 *   OpenMP 32 threads ..... ______ s   speedup ____
 *   CUDA .................. 2.685 s   speedup 9.66
 *
 *  Compilar: gcc -O2 -std=c11 -Wall -o kmeans_seq kmeans_seq.c -lm
 *  Executar: ./kmeans_seq data/covtype.bin 32 300
 * =====================================================================
 */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <time.h>

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


// Paralelizar 1 - Motivo: Cada ponto i é processado de forma independente
// Cada ponto lê X e C, que ninguém altera
// Escreve só no seu próprio labels[i]
// Problema: mudanca pode haver c.c
static long atribuir_pontos(const float *X, const float *C, int *labels,
                            int n, int d, int k) {

    //para cada ponto, descobre qual centróide está mais perto e grava o número dele em labels[i]

    // Contamos as mudanças pois é o critério de parada. 
    // Se nenhum ponto trocou de cluster, os centróides também não vão mudar na atualização, e o algoritmo convergiu
    long mudancas = 0;
    for (int i = 0; i < n; i++) {
        const float *p = &X[(size_t)i * d];
        int melhor = 0;
        //começa com o maior float possível
        float melhor_dist = FLT_MAX;

        //compara o ponto com os K centróides e guarda o índice do mais próximo em melhor
        for (int c = 0; c < k; c++) {
            float dd = distancia_quadrada(p, &C[(size_t)c * d], d);
            if (dd < melhor_dist) {
                melhor_dist = dd;
                melhor = c;
            }
        }

        //se o ponto mudou de cluster, atualiza o rótulo e conta uma mudança.
        if (labels[i] != melhor) {
            labels[i] = melhor;
            mudancas++;
        }
    }

    //No fim, devolve quantos pontos trocaram de cluster.
    return mudancas;
}


// Paralelizar 2 - Motivo: o laço de soma percorre os N pontos (O(N*D)),
// e, pela Lei de Amdahl, deixar sequencial limitaria o speedup com muitas threads
// Problema: vários pontos do mesmo cluster escrevem no mesmo somas[c] e contagens[c] -> c.c
// Cada centróide passa a ser a média dos pontos que pertencem ao cluster dele
static void atualizar_centroides(const float *X, float *C, const int *labels,
                                 double *somas, long *contagens, int n, int d, int k) {

    //OBS: Soma em float não é exatamente associativa, mudar a ordem muda levemente o resultado.

    //zerando acumuladores
    memset(somas, 0, (size_t)k * d * sizeof(double)); //para cada cluster, faz a soma de cada atributo dos seus pontos
    memset(contagens, 0, (size_t)k * sizeof(long)); //quantos pontos cada cluster tem.

    // PARTE PARALELIZÁVEL
    //Para cada ponto, pega o cluster dele, soma os atributos e incrementa contagens
    for (int i = 0; i < n; i++) {
        int c = labels[i];
        const float *p = &X[(size_t)i * d];
        double *s = &somas[(size_t)c * d];
        for (int j = 0; j < d; j++) s[j] += p[j];
        contagens[c]++;
    }

    //Média = soma ÷ quantidade, atributo por atributo
    for (int c = 0; c < k; c++) {
        // Se nenhum ponto caiu num cluster, contagens é 0 e a divisão daria erro. 
        // O continue mantém o centróide onde estava.
        if (contagens[c] == 0) continue;
        for (int j = 0; j < d; j++) {
            C[(size_t)c * d + j] = (float)(somas[(size_t)c * d + j] / contagens[c]);
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

    float  *C         = malloc((size_t)k * d * sizeof(float));
    int    *labels    = malloc((size_t)n * sizeof(int));
    double *somas     = malloc((size_t)k * d * sizeof(double));
    long   *contagens = malloc((size_t)k * sizeof(long));
    if (!C || !labels || !somas || !contagens) {
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
        atualizar_centroides(X, C, labels, somas, contagens, n, d, k);
    }
    double t_kmeans = tempo_atual() - t0;

    double sse = calcular_sse(X, C, labels, n, d);

    printf("Versao       : sequencial\n");
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
    return EXIT_SUCCESS;
}
