require 'torch'   -- torch
require 'gfx.js'  -- to visualize the dataset
require 'nn'      -- provides a normalization operator
local Data = require 'NeuralNet.utils.Data'

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Dataset Preprocessing')
   cmd:text()
   cmd:text('Options:')      
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
print '==> loading train dataset'

local trainLoaded = Data.fileToTensor{file="/home/pawel/dev/data/FX/truefx/2014/1/AUDJPY-20140102",nColumns=11,sep=","}

trainData = {
   data = trainLoaded[{{},{2,11}}],
   labels = trainLoaded[{{},1}],
   size = function() return (#trainLoaded)[1] end
}

print(string.format("Loaded train data: %s",trainLoaded:size()))

print '==> loading test dataset'

-- Finally we load the test data.
local testLoaded = Data.fileToTensor{file="/home/pawel/dev/data/FX/truefx/2014/1/AUDJPY-20140103",nColumns=11,sep=","}

testData = {
   data = testLoaded[{{},{2,11}}],
   labels = testLoaded[{{},1}],
   size = function() return (#testLoaded)[1] end
}

print(string.format("Loaded test data: %s",testLoaded:size()))


-- Normalize each channel, and store mean/std
-- per channel. These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.
print '==> preprocessing data: normalize each feature globally'
mean = torch.mean(trainData.data,1)
std = torch.std(trainData.data,1)

trainData.data:add(-mean:repeatTensor(trainData.data:size(1),1))
trainData.data:cdiv(std:repeatTensor(trainData.data:size(1),1))

testData.data:add(-mean:repeatTensor(testData.data:size(1),1))
testData.data:cdiv(std:repeatTensor(testData.data:size(1),1))

----------------------------------------------------------------------
print '==> verify statistics'

-- It's always good practice to verify that data is properly
-- normalized.

trainMean = torch.mean(trainData.data, 1)
trainStd = torch.std(trainData.data, 1)

testMean = torch.mean(testData.data,1)
testStd = torch.std(testData.data,1)

print(string.format('training data, mean: %s',trainMean))
print(string.format('training data, standard deviation: %s',trainStd))

print(string.format('test data, mean: %s',testMean))
print(string.format('test data, standard deviation: %s',testStd))
