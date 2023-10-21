n = 0.5; % Passo de aprendizagem
W = -0.5 +  rand(1, 3); % Vetor de pesos aleatórios

% Vetor de entradas
X = [ 
    [-1 0 0];
    [-1 0 1];
    [-1 1 0];
    [-1 1 1];
];



% Calcula os pesos resultantes do treinamento
resultsWeights = train(10, W, X, n);

% Plota o gráfico do hiperplano
plotGraph(resultsWeights);


% Função para treinar o neurônio com base nos pesos aleatórios iniciais 
% 'W', o vetor de entradas 'X', o passo de aprendizagem 'n' e a quantidade
% de Épocas adequada.
function adjustedW = train(epochs, W, X, n)
    sameW = 0;

    for i = 1:epochs
        
        for j = 1:length(X)
        
            input = X(j, :);
           
            [~ , e] = calculateActivation(W, input);
            
            
            WUpdated = updateWeight(W, input, n, e);
            
            if isequal(WUpdated, W)
                sameW = sameW + 1;
            else
                sameW = 0;
            end

            W = WUpdated;

            if sameW > 6
                adjustedW = W;
                return;
            end

        end

    end
    
    adjustedW = [0, 0, 0];
    disp('Finalizou a quantidade de épocas sem estabilizar os pesos!')
end

% Calcula o saída e o erro gerado com base nos pesos atuais 'W' e a 
% entrada atual escolhida 'X'
function [output, error] = calculateActivation(W, X)
    outputY = 0;                    %Saída
    desireValue = X(2) || X(3);     %Saída desejada
    
    activation = sum(W .* X); %Calculo da ativação u
    
    % Função degrau de ativação do neurônio:
    if activation > 0
        outputY = 1;
    end

    e = desireValue - outputY; %Erro
    
    output = outputY;
    error = e;

end


% Função para atualizar os pesos com base no peso anterior 'W', a saída
% 'X', o passo de aprendizado 'n' e o erro gerado
function updatedW = updateWeight(W, X, n, error)   
    updatedW = W + (n*error).*X;    
end


% Plotagem do gráfico com as classes do problema e o hiperplano que divide
% as classes True e False. De acordo com os pesos resultantes após o
% treinamento do neurônio.
function plotGraph(W)
       
    % Plotagem do gráfico
    figure;
    hold on;
    scatter(0, 0, 100, 'x', 'r', 'MarkerEdgeColor', 'r', 'LineWidth', 3);
    scatter(0, 1, 100, 'o', 'filled', 'g', 'LineWidth', 2); 
    scatter(1, 0, 100, 'o', 'filled', 'g', 'LineWidth', 2); 
    scatter(1, 1, 100, 'o', 'filled', 'g', 'LineWidth', 2); 
    
    xlabel('X1');
    ylabel('X2');
    title('Porta Lógica OR - Classificação pelo Perceptron');
    legend('Classe False', 'Classe True');
    grid on;
    
    % Desenhe a reta de decisão (hiperplano) usando os pesos resultantes
    x = -1:0.01:2; % Valores de x para a plotagem
    y = -(W(2) / W(3)) * x + (W(1) / W(3));
    
    % Desenha a reta no gráfico
    plot(x, y, 'b', 'LineWidth', 3, 'DisplayName', 'Hiperplano (reta de decisão)');

    hold off;


end







    