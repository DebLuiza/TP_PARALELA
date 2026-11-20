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


 /*
 PODE CORTAR ESSA PARTE, DEIXA SÓ O CABEÇALHO
 * =====================================================================
 *  PARTES PARALELIZÁVEIS
 * =====================================================================
 *
 *  1) atribuir_pontos() - laço sobre os N pontos (i)
 *     - Parte mais cara do programa: O(N*K*D) por iteração.
 *     - Cada ponto é processado de forma independente: lê X e C (que
 *       ninguém altera) e escreve só no seu próprio labels[i].
 *     - Variável compartilhada: mudancas -> condição de corrida.
 *
 *  2) atualizar_centroides() - laço de soma sobre os N pontos (i)
 *     - Custo O(N*D) por iteração; menor que a atribuição, mas, pela
 *       Lei de Amdahl, deixá-lo sequencial limitaria o speedup.
 *     - Vários pontos do mesmo cluster escrevem no mesmo somas[c] e
 *       contagens[c] -> condição de corrida.
 *     - O laço final de divisão (K*D = 1.728 operações) fica sequencial.
 *
 * =====================================================================
 *  POSSÍVEIS PERGUNTAS
 * =====================================================================
 *
 *  P: Por que paralelizar o laço i e não o laço c na atribuição?
 *  R: O laço i tem N = 581 mil iterações; o c tem só K = 32 e ainda
 *     exigiria sincronização para achar o mínimo. Paralelizar o laço
 *     externo dá mais trabalho por thread e menos sincronização.
 *
 *  P: Por que não paralelizar o laço de iterações do K-means (while)?
 *  R: Há dependência: a iteração t+1 precisa dos centróides calculados
 *     na iteração t. O paralelismo fica dentro de cada iteração.
 *
 *  P: Onde há condição de corrida?
 *  R: Na contagem de mudancas (atribuição) e nos acumuladores somas e
 *     contagens (atualização), escritos por vários pontos ao mesmo tempo.
 *
 *  P: Se a atribuição é a parte cara, por que paralelizar a atualização?
 *  R: Lei de Amdahl: com K = 32, a soma é cerca de 3% do tempo. Com 32
 *     threads, deixar esses 3% sequenciais limitaria o speedup a ~16x
 *     em vez de ~32x.
 *
 *  P: Os resultados paralelos são idênticos ao sequencial?
 *  R: Quase. A soma em ponto flutuante não é associativa, e a ordem das
 *     somas muda na versão paralela. Por isso somas é double: a diferença
 *     fica só nas últimas casas decimais.
 *
 *  P: Como vocês validaram as versões paralelas?
 *  R: Inicialização determinística (centróide c = ponto c*N/K): todas as
 *     versões partem dos mesmos centróides e devem chegar ao mesmo número
 *     de iterações (34), ao mesmo SSE e aos mesmos tamanhos de cluster.
 *
 *  P: Por que o tempo de leitura não entra na medição?
 *  R: É I/O de disco, não faz parte do algoritmo e não é paralelizável;
 *     incluí-lo esconderia o speedup.
 *
 *  P: Por que usar a distância ao quadrado, sem raiz?
 *  R: A raiz é crescente, então não muda qual centróide é o mais próximo,
 *     e evita N*K chamadas de sqrt por iteração.
 *
 *  P: O K-means sempre converge?
 *  R: Sim, para um mínimo local: o SSE nunca aumenta entre iterações e o
 *     número de agrupamentos possíveis é finito. O resultado depende da
 *     inicialização (o K-means++ melhora isso).
 * =====================================================================
 */