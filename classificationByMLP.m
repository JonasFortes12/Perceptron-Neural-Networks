% Classes
% [1 0 0] - DH(disk hernia);
% [0 1 0] - SL(spondilolysthesis);
% [0 0 1] - NO(Normal)

data = load("./data/column_3C.dat");

% Adequando os rótulos de dados no formato 'one-hot'
data(1:60, 7:9) = repmat([1 0 0], length(1:60), 1);
data(61:210, 7:9) = repmat([0 1 0], length(61:210), 1);
data(211:310, 7:9) = repmat([0 0 1], length(211:310), 1);

% Normaliza os dados 
data = normalizeData(data);

% Vetor de acurácias
accuracies = [];

% Realiza 10 iterações de classificação dos dados
for i = 1:10

    % Permuta os dados
    swappedData = exchangeData(data);
    
    % hold-out (70% das amostras para treino e o restante para teste)
    dataTrain = swappedData(1:217, :)'; 
    dataTest = swappedData(218:310, :)';
    
    % Separa os vetores de características dos vetores de rótulos para os dados
    % de treino e de teste
    XTrain = dataTrain(1:6,:);
    YTrain = dataTrain(7:9,:);
    XTest = dataTest(1:6,:);
    YTest = dataTest(7:9,:);
    
    % Instancia a rede neural MLP
    net = feedforwardnet(10);
    
    % Realiza o treinamento da rede
    net = train(net, XTrain, YTrain);

    % Rede MLP classifica os dados de teste e retorna os rótulos
    Y = net(XTest);
    
    % Adiciona ao vetor de acurácias
    accuracies = [accuracies  calculateAccuracy(Y, YTest)];

end

fprintf('Acurácia Média: %.2f%%\n', sum(accuracies)/length(accuracies));


% Função para permutar os dados
function swappedData = exchangeData(data)
    swappedData = data(randperm(size(data, 1)), :);
end

% Calcula a acurácia da classificação da rede MLP (0 a 100)%
function accuracy = calculateAccuracy(Y, YTest)
    
    [~ , indexMaxYTest] = max(YTest);
    [~ , indexMaxY]     = max(Y);
    
    % Quantidades de acertos
    hits = sum(indexMaxYTest == indexMaxY);
    
    accuracy = (hits / 93) * 100;
    
end

% Função para normalizar os dados
function normalizedData = normalizeData(data)
    normalizedData = (data - min(data)) ./ (max(data) - min(data));
end



