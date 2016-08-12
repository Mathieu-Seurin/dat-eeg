-- Import
require 'torch'
local matio = require 'matio'
require 'nn'
require 'cunn'
require 'cunnx'
require 'cutorch'
require 'optim'
require 'xlua'

--[[ 
   Function To Call
   Here :
--]]

function SplitTrainTest(X,y,split)
   split = split or 0.8
   
   cardExemple = y:size(1)
   randomTensor = torch.randperm(cardExemple):long()
   sizeSplit = math.floor(split*cardExemple)
   
   posIndex = randomTensor[{{1,sizeSplit}}]
   negIndex = randomTensor[{{sizeSplit+1,randomTensor:size(1)}}]

   xTrain = X:index(1,posIndex)
   xTest = X:index(1,negIndex)
   yTrain = y:index(1,posIndex)
   yTest = y:index(1,negIndex)

   return xTrain,xTest,yTrain,yTest
end
-- Loading data ===========================================
data = matio.load('../Data/CFiltered.mat')

X = data['X']
y = data['y']
y = y:reshape(y:size(2))

y[y:eq(1)] = 2
y[y:eq(-1)] = 1

cardExemple = X:size(1)
cardFeatures = X:size(2)

xTrain,xTest,yTrain,yTest = SplitTrainTest(X,y)
X, y = nil,nil
trSize = math.floor(cardExemple*0.80)

trainData = {
   data = xTrain:float(),
   labels = yTrain:int(),
   size = function () return (#trainData.data)[1] end
}

testData = {
   data = xTest:float(),
   labels = yTest:int(),
   size = function () return (#testData.data)[1] end
}

ninputs = xTrain:size(2)

nhidden = 100
noutputs = 2

model = nn.Sequential()

model:add(nn.View(ninputs))
model:add(nn.Linear(ninputs,nhidden))
model:add(nn.Tanh())
model:add(nn.Linear(nhidden,noutputs))
--model:add(nn.LogSoftMax())

criterion = nn.MultiMarginCriterion()
--criterion = nn.ClassNLLCriterion()

model:cuda()
criterion:cuda()

-- TOOLS ===========================================

classes = {'0','1'}
confusion = optim.ConfusionMatrix(classes)

trainLogger = optim.Logger(paths.concat('Results/', 'train.log'))
testLogger = optim.Logger(paths.concat('Results/', 'test.log'))

parameters, gradParameters = model:getParameters()

optimState = {
   learningRate = 1e-3,
   weightDecay = 0,
   momentum = 0,
   learningRateDecay = 1e-7
}

optimMethod = optim.sgd

function train()
   epoch = epoch or 1
   local time = sys.clock()

   model:training()

   shuffle = torch.randperm(trSize)

   batchSize = 100
   
   for t=1,trainData:size(),batchSize do

      xlua.progress(t, trainData:size())

      local inputs = {}
      local targets = {}

      for i=t,math.min(t+batchSize-1,trainData:size()) do
         local input = trainData.data[shuffle[i]]:cuda()
         local target = trainData.labels[shuffle[i]]
         table.insert(inputs, input)
         table.insert(targets, target)
      end

      local feval = function(x)
         if x ~= parameters then
            parameters:copy(x)
         end
         
         gradParameters:zero()
         
         local f=0

         for i=1,#inputs do
            local output = model:forward(inputs[i])
            local err = criterion:forward(output,targets[i])
            f = f+err

            local df_do = criterion:backward(output,targets[i])
            model:backward(inputs[i],df_do)

            confusion:add(output,targets[i])
         end
         
         gradParameters:div(#inputs)
         f = f/#inputs

         return f,gradParameters
      end
      
      optimMethod(feval, parameters, optimState)    
   end


   time = sys.clock() - time
   time = time / trainData:size()
   print("==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   torch.save('model.net', model)

   print(confusion)
   confusion:zero()

   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid}


   local filename = paths.concat('Results/', 'model.net')
   print("Saving Model")
   torch.save(filename,model)

   epoch = epoch+1
end


function test()
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')
   for t = 1,testData:size() do
      -- disp progress
      xlua.progress(t, testData:size())

      -- get new sample
      local input = testData.data[t]:cuda()
      
      local target = testData.labels[t]

      -- test sample
      local pred = model:forward(input)
      confusion:add(pred, target)
   end

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
   -- next iteration:
   confusion:zero()
end


while true do
   train()
   test()
end
