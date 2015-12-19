----------------------------------------------------------------------
-- This script demonstrates how to define a couple of different
-- models:
--   + linear
--   + 2-layer neural network (MLP)
--   + convolutional network (ConvNet)
--
-- It's a good idea to run this script with the interactive mode:
-- $ torch -i 2_model.lua
-- this will give you a Torch interpreter at the end, that you
-- can use to play with the model.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- for image transforms
require 'gfx.js'  -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Model Definition')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-model', 'mlp', 'type of model to construct: linear | mlp')
   cmd:option('-noutputs', 2 , 'number of outputs')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
print '==> define parameters'

-- 2-class problem
noutputs = opt.noutputs

-- input dimensions
ninputs = trainData.data:size(2)

-- number of hidden units (for MLP only):
nhiddens = ninputs / 2

----------------------------------------------------------------------
print('==> construct model: '..opt.model)

if opt.model == 'linear' then

   -- Simple linear model
   model = nn.Sequential()   
   model:add(nn.Linear(ninputs,noutputs))

elseif opt.model == 'mlp' then

   -- Simple 2-layer neural network, with tanh hidden units
   model = nn.Sequential()   
   model:add(nn.Linear(ninputs,nhiddens))
   model:add(nn.Tanh())
   model:add(nn.Linear(nhiddens,noutputs))

else

   error('unknown -model')

end

----------------------------------------------------------------------
print '==> here is the model:'
print(model)

