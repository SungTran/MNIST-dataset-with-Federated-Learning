%% Start pool environment and assign numWorkers = 5
cluster = parcluster("Processes");
cluster.NumWorkers = 5;
pool = parpool(cluster);
%pool = parpool('local',5);
numWorkers = pool.NumWorkers;
%% Load dataset into workers
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos',...
    'nndatasets','DigitDataset');
inputSize = [28 28 1];
spmd   
    digitDatasetPath1 = fullfile(digitDatasetPath,num2str(labindex - 1));
    imds1 = imageDatastore(digitDatasetPath1,...
        'IncludeSubfolders',true,...
        'LabelSource','foldernames');
    digitDatasetPath2 = fullfile(digitDatasetPath,num2str(labindex + 4));
    imds2 = imageDatastore(digitDatasetPath2,...
        'IncludeSubfolders',true,...
        'LabelSource','foldernames');
    imds = imageDatastore({digitDatasetPath2,digitDatasetPath1},...
        'IncludeSubfolders',true,...
        'LabelSource','foldernames');
    [imdsTrain,imdsTestVal] = splitEachLabel(imds,0.7,"randomized");
    
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
end
%% Concatenate files and Labels of Testing and Validating data from all workers
%to form global testing data
fileList = [];
labelList = [];

for i = 1:numWorkers
    tmp = imdsTestVal{i};
    
    fileList = cat(1,fileList,tmp.Files);
    labelList = cat(1,labelList,tmp.Labels);    
end

imdsGlobalTestVal = imageDatastore(fileList);
imdsGlobalTestVal.Labels = labelList;

[imdsGlobalTest,imdsGlobalVal] = splitEachLabel(imdsGlobalTestVal,0.5,"randomized");

augimdsGlobalTest = augmentedImageDatastore(inputSize(1:2),imdsGlobalTest);
augimdsGlobalVal = augmentedImageDatastore(inputSize(1:2),imdsGlobalVal);
%% Define network initial structure for all models

classes = categories(imdsGlobalTest.Labels);
numClasses = numel(classes);
layers = [
    imageInputLayer(inputSize,'Normalization','none','Name','input')
    convolution2dLayer(5,32,'Name','conv1')
    reluLayer('Name','relu1')
    maxPooling2dLayer(2,'Name','maxpool1')
    convolution2dLayer(5,64,'Name','conv2')
    reluLayer('Name','relu2')
    maxPooling2dLayer(2,'Name','maxpool2')
    fullyConnectedLayer(numClasses,'Name','fc')
    softmaxLayer('Name','softmax')];
net = dlnetwork(layers);
numRounds = 100;
numEpochsperRound = 5;
miniBatchSize = 100;
learnRate = 0.001;
momentum = 0;
%% convert the labels to one-hot encoded variables
%converts the data to dlarray objects with underlying type single
preProcess = @(x,y)preprocessMiniBatch(x,y,classes);

spmd
    sizeOfLocalDataset = augimdsTrain.NumObservations;
    
    mbq = minibatchqueue(augimdsTrain,...
        'MiniBatchSize',miniBatchSize,...
        'MiniBatchFcn',preProcess,...
        'MiniBatchFormat',{'SSCB',''});
end
%% Create a minibatchqueue object that manages the validation data to use 
% during training. Use the same settings as the minibatchqueue on each worker.

mbqGlobalVal = minibatchqueue(augimdsGlobalVal,...
    'MiniBatchSize',miniBatchSize,...
    'MiniBatchFcn',preProcess,...
    'MiniBatchFormat',{'SSCB',''});
%%Plot the training process 
figure
lineAccuracyTrain = animatedline('Color',[0.85 0.325 0.098]);
ylim([0 inf])
xlabel("Communication rounds")
ylabel("Accuracy (Global)")
grid on
velocity = [];
%% Initialize the model
globalModel = net;
start = tic;
for rounds = 1:numRounds
   
    spmd
        % Send global updated parameters to each worker.
        net.Learnables.Value = globalModel.Learnables.Value;        
        
        % Loop over epochs.
        for epoch = 1:numEpochsperRound
            % Shuffle data.
            shuffle(mbq);
            
            % Loop over mini-batches.
            while hasdata(mbq)
                
                % Read mini-batch of data.
                [X,T] = next(mbq);
                
                % Evaluate the model loss and gradients using dlfeval and the
                % modelLoss function.
                [loss,gradients] = dlfeval(@modelLoss,net,X,T);
                
                % Update the network parameters using the SGDM optimizer.
                [net,velocity] = sgdmupdate(net,gradients,velocity,learnRate,momentum);
                
            end
        end
        
        % Collect updated learnable parameters on each worker.
        workerLearnables = net.Learnables.Value;
    end
    
    % Find normalization factors for each worker based on ratio of data
    % processed on that worker. 
    sizeOfAllDatasets = sum([sizeOfLocalDataset{:}]);
    normalizationFactor = [sizeOfLocalDataset{:}]/sizeOfAllDatasets;
    
    % Update the global model with new learnable parameters, normalized and
    % averaged across all workers.
    globalModel.Learnables.Value = federatedAveraging(workerLearnables,normalizationFactor);
    
    % Calculate the accuracy of the global model.
    accuracy = computeAccuracy(globalModel,mbqGlobalVal,classes);
    
    % Display the training progress of the global model.
    D = duration(0,0,toc(start),'Format','hh:mm:ss');
    addpoints(lineAccuracyTrain,rounds,double(accuracy))
    title("Communication round: " + rounds + ", Elapsed: " + string(D))
    drawnow
end
%% update the network on each worker with the final average learnable parameters
spmd
    net.Learnables.Value = globalModel.Learnables.Value;
end
%% Test the model
mbqGlobalTest = minibatchqueue(augimdsGlobalTest,...
    'MiniBatchSize',miniBatchSize,...
    'MiniBatchFcn',preProcess,...
    'MiniBatchFormat','SSCB');
accuracy = computeAccuracy(globalModel,mbqGlobalTest,classes)
delete(gcp('nocreate'));
%% Model Loss Function
function [loss,gradients] = modelLoss(net,X,T)

    YPred = forward(net,X);
    
    loss = crossentropy(YPred,T);
    gradients = dlgradient(loss,net.Learnables);

end
%% Compute Accuracy Function
function accuracy = computeAccuracy(net,mbq,classes)

    correctPredictions = [];
    
    shuffle(mbq);
    while hasdata(mbq)
        
        [XTest,TTest] = next(mbq);
        
        TTest = onehotdecode(TTest,classes,1)';
        
        YPred = predict(net,XTest);
        YPred = onehotdecode(YPred,classes,1)';
        
        correctPredictions = [correctPredictions; YPred == TTest];
    end
    
    predSum = sum(correctPredictions);
    accuracy = single(predSum./size(correctPredictions,1));

    end
    %% Mini-Batch Preprocessing Function
function [X,Y] = preprocessMiniBatch(XCell,YCell,classes)

    % Concatenate.
    X = cat(4,XCell{1:end});
    
    % Extract label data from cell and concatenate.
    Y = cat(2,YCell{1:end});
    
    % One-hot encode labels.
    Y = onehotencode(Y,1,'ClassNames',classes);

    end
    %% Federated Averaging Function
function learnables = federatedAveraging(workerLearnables,normalizationFactor)

    numWorkers = size(normalizationFactor,2);
    
    % Initialize container for averaged learnables with same size as existing
    % learnables. Use learnables of first worker network as an example.
    exampleLearnables = workerLearnables{1};
    learnables = cell(height(exampleLearnables),1);
    
    for i = 1:height(learnables)   
        learnables{i} = zeros(size(exampleLearnables{i}),'like',(exampleLearnables{i}));
    end
    
    % Add the normalized learnable parameters of all workers to
    % calculate average values.
    for i = 1:numWorkers
        tmp = workerLearnables{i};
        for values = 1:numel(learnables)
            learnables{values} = learnables{values} + normalizationFactor(i).*tmp{values};
        end
    end
    
end