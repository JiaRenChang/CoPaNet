--
-- Competitive Pathway Network (CoPaNet)
-- 

local nn = require 'nn'
require 'cunn'


local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
   local depth = opt.depth
   local shortcutType = opt.shortcutType or 'B'
   local iChannels

   -- The shortcut layer is either identity or 1x1 convolution
   local function shortcut(nInputPlane, nOutputPlane, stride)
      local useConv = shortcutType == 'C' or
         (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
      if useConv then
         -- 1x1 convolution
         return nn.Sequential()
            :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
            --:add(SBatchNorm(nOutputPlane))
      elseif nInputPlane ~= nOutputPlane then
         -- Strided, zero-padded identity shortcut
         return nn.Sequential()
            :add(nn.SpatialAveragePooling(1, 1, stride, stride))
            :add(nn.Padding(1, (nOutputPlane - nInputPlane), 3))
      else
         return nn.Identity()
      end
   end

   -- The basic layer
   local function basicblock(inch,outch, stride)
      local nInputPlane = inch
      n = outch
      local s = nn.Sequential()

      s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1)) 
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,1,1,1,1)) 

   
      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n, stride)))
         :add(nn.CAddTable(true)) 
    
   end

   -- The bottleneck layer
   local function bottleneck(inch,outch, stride,i,count)
      local nInputPlane = inch
      local n = outch/4
      local s = nn.Sequential()

      s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n)) 
      s:add(ReLU(true))
      s:add(Convolution(n,outch,1,1,1,1,0,0))


      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, outch, stride)))
         :add(nn.CAddTable(true))
   end



   -- Creates count residual blocks with specified number of features
   local function layer(block, infeatures, outfeatures, count, stride)
      local s = nn.Sequential()
      for i=1,count do

         s:add(nn.ConcatTable()
              :add(block(i ==1 and infeatures or outfeatures ,outfeatures, i == 1 and stride or 1,i,count))
              :add(block(i ==1 and infeatures or outfeatures ,outfeatures, i == 1 and stride or 1,i,count)))
         s:add(nn.CMaxTable())

      end
      return s
   end

   local model = nn.Sequential()
   if opt.dataset == 'imagenet' then
      print('Not finished')

   elseif opt.dataset == 'cifar10' then
      -- Model type specifies number of layers for CIFAR-10 model
      assert((depth - 2) % 9 == 0, 'depth should be one of 20, 32, 44, 56, 110, 1202')
      local n = (depth - 2) / 9
      --iChannels = 16*4
      print(' | CoPaNet-' .. depth .. ' CIFAR-10' .. ' layers:' .. n)
      local wide = 1*4
      -- The CIFAR-10 model
      model:add(Convolution(3,16,3,3,1,1,1,1))
      model:add(SBatchNorm(16))
      model:add(ReLU(true))
      
      model:add(layer(bottleneck, 16,  12*wide, n, 1))
      model:add(SBatchNorm(12*wide))
      model:add(ReLU(true))
      model:add(Avg(2,2,2,2))
      model:add(nn.Dropout(0.2))

      concateF = nn.Concat(2)
      concateF:add(nn.Identity())
      concateF:add(layer(bottleneck, 12*wide,  24*wide, n, 1))
      model:add(concateF)
      model:add(SBatchNorm(36*wide))  
      model:add(ReLU(true))
      model:add(Avg(2,2,2,2))       
      model:add(nn.Dropout(0.2))

      concateS = nn.Concat(2)
      concateS:add(nn.Identity()) 
      concateS:add(layer(bottleneck, 36*wide,  45*wide, n, 1))
      model:add(concateS)
      model:add(SBatchNorm(81*wide))  
      model:add(ReLU(true))
      model:add(Avg(8, 8, 1, 1))

      model:add(nn.View(81*wide):setNumInputDims(3)) 
      model:add(nn.Linear(81*wide, 10))
 
   elseif opt.dataset == 'cifar100' then
      -- Model type specifies number of layers for CIFAR-10 model
      assert((depth - 2) % 9 == 0, 'depth should be one of 20, 32, 44, 56, 110, 1202')
      local n = (depth - 2) / 9
      --iChannels = 16*4
      print(' | CoPaNet-' .. depth .. ' CIFAR-100' .. ' layers:' .. n)
      local wide = 1*4

      -- The CoPaNet CIFAR-100 model
      model:add(Convolution(3,16,3,3,1,1,1,1))
      model:add(SBatchNorm(16))
      model:add(ReLU(true))
      
      model:add(layer(bottleneck, 16,  12*wide, n, 1))
      model:add(SBatchNorm(12*wide))
      model:add(ReLU(true))
      model:add(Avg(2,2,2,2))
      model:add(nn.Dropout(0.2))

      concateF = nn.Concat(2)
      concateF:add(nn.Identity()) 
      concateF:add(layer(bottleneck, 12*wide,  24*wide, n, 1))
      model:add(concateF)
      model:add(SBatchNorm(36*wide))  
      model:add(ReLU(true))
      model:add(Avg(2,2,2,2))       
      model:add(nn.Dropout(0.2))

      concateS = nn.Concat(2)
      concateS:add(nn.Identity()) 
      concateS:add(layer(bottleneck, 36*wide,  45*wide, n, 1))
      model:add(concateS)
      model:add(SBatchNorm(81*wide))
      model:add(ReLU(true))
      model:add(Avg(8, 8, 1, 1))

      model:add(nn.View(81*wide):setNumInputDims(3)) 
      model:add(nn.Linear(81*wide, 100))

   elseif opt.dataset == 'svhn' then
      -- Model type specifies number of layers for CIFAR-10 model
      assert((depth - 2) % 9 == 0, 'depth should be one of 20, 32, 44, 56, 110, 1202')
      local n = (depth - 2) / 9
      --iChannels = 16*4
      print(' | CoPaNet-' .. depth .. ' svhn' .. ' layers:' .. n)
      local wide = 1*4

      model:add(Convolution(3,16,3,3,1,1,1,1))
      model:add(SBatchNorm(16))
      model:add(ReLU(true))
      
      model:add(layer(bottleneck, 16,  12*wide, n, 1))
      model:add(SBatchNorm(12*wide))
      model:add(ReLU(true))
      model:add(Avg(2,2,2,2))
      model:add(nn.Dropout(0.2))

      concateF = nn.Concat(2)
      concateF:add(nn.Identity()) 
      concateF:add(layer(bottleneck, 12*wide,  24*wide, n, 1))
      model:add(concateF)
      model:add(SBatchNorm(36*wide))  
      model:add(ReLU(true))
      model:add(Avg(2,2,2,2))       
      model:add(nn.Dropout(0.2))

      concateS = nn.Concat(2)
      concateS:add(nn.Identity()) 
      concateS:add(layer(bottleneck, 36*wide,  45*wide, n, 1))
      model:add(concateS)
      model:add(SBatchNorm(81*wide))
      model:add(ReLU(true))
      model:add(Avg(8, 8, 1, 1))

      model:add(nn.View(81*wide):setNumInputDims(3)) 
      model:add(nn.Linear(81*wide, 10))
      
   else
      error('invalid dataset: ' .. opt.dataset)
   end

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:cuda()

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

      print(model)
      print(model:getParameters():size(1))
   return model
end

return createModel
