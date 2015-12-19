--require 'cutorch'
--torch.setdefaulttensortype('torch.CudaTensor')

--require 'torch'
--torch.setnumthreads(11)
--
--timer = torch.Timer()
--
--t1 = torch.rand(10000,10000)
--t2 = torch.rand(10000,10000)
--res = torch.Tensor(10000,10000):fill(0)
--
--for i = 1,10 do
--  res:mm(t1,t2)
--end
--print(res:max())
--print('Time: ' .. timer:time().real .. ' seconds')

plp = require 'pl.pretty'
require "torch"

r=torch.Tensor{2,2,2}
m=torch.Tensor{{1,2,3},{4,5,6}}
print(m:cdiv(r:repeatTensor(2,1)))



