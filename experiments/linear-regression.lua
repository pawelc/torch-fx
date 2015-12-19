--require 'torch'
--require 'nn'

require 'cunn'
require 'cutorch'
torch.setdefaulttensortype('torch.CudaTensor')

require 'optim'
local Data = require 'NeuralNet.utils.Data'

--torch.setnumthreads(12)
--Using liner regression to predict direction of the next tick

local data = Data.fileToTensor{file="/home/pawel/dev/data/FX/truefx/2014/1/AUDJPY-train.txt",nColumns=11,sep=","}

model = nn.Sequential()      

ninputs = 10; noutputs = 1
model:add(nn.Linear(ninputs, noutputs))

criterion = nn.MSECriterion()

model:cuda()
criterion:cuda()
 
x, dl_dx = model:getParameters()

--print(string.format("Original: %s",x))

feval = function(x_new)   
   if x ~= x_new then
      x:copy(x_new)
   end

   _nidx_ = (_nidx_ or 0) + 1
   if _nidx_ > (#data)[1] then _nidx_ = 1 end

   local sample = data[_nidx_]
   local target = sample[{ {1} }]
   local inputs = sample[{ {2,11} }]
   
   dl_dx:zero()

   local loss_x = criterion:forward(model:forward(inputs), target)
   model:backward(inputs, criterion:backward(model.output, target))

   return loss_x, dl_dx
end

sgd_params = {
   learningRate = 1e-5,
   learningRateDecay = 1e-4,
   weightDecay = 1e-3,
   momentum = 0
}

timer = torch.Timer()
for i = 1,100 do
   current_loss = 0

   for i = 1,(#data)[1] do      
      _,fs = optim.sgd(feval,x,sgd_params)
      current_loss = current_loss + fs[1]
   end

   current_loss = current_loss / (#data)[1]
--   print('current loss = ' .. current_loss)
end
print('Training time: ' .. timer:time().real .. ' seconds')

--print(string.format("Weights: %s",x))     

local testData = data
local accuracy = 0
for i = 1,(#testData)[1] do
  res = model:forward(testData[{{i},{2,11}}])[1][1] 
  if res > 0 then
  print('positive')
  end
  
   if model:forward(testData[{{i},{2,11}}])[1][1] * testData[{i,1}] > 0 then
    accuracy = accuracy + 1
   end    
end
accuracy = accuracy/(#testData)[1]

print("Accuracy: "..accuracy)

 