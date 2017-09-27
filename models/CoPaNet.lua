--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The ResNet model definition
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
            :add(SBatchNorm(nOutputPlane))
      elseif nInputPlane ~= nOutputPlane then
         -- Strided, zero-padded identity shortcut
         return nn.Sequential()
            :add(nn.SpatialAveragePooling(1, 1, stride, stride))
            :add(nn.Padding(1, (nOutputPlane - nInputPlane), 3))
      else
         return nn.Identity()
      end
   end

   local function ShareGradInput(module, key)
      assert(key)
      module.__shareGradInputKey = key
      return module
   end

   -- The basic residual layer block for 18 and 34 layer network, and the
   -- CIFAR networks
   local function basicblock(inch,outch, stride)
      local nInputPlane = inch
      n = outch


      local s = nn.Sequential()

      s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1)) 
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,1,1,1,1)) 

   
      return nn.Sequential()
         :add(ShareGradInput(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n, stride)),'basicconcat'))
         :add(nn.CAddTable(true)) 
    
   end

   -- The bottleneck residual layer for 50, 101, and 152 layer networks
   local function bottleneck(inch,outch, stride,i,count)
      local nInputPlane = inch
      local n = outch/4
      local block = nn.Sequential()
      local s = nn.Sequential()

      s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n)) 
      s:add(ReLU(true))
      s:add(Convolution(n,outch,1,1,1,1,0,0))
      s:add(SBatchNorm(outch)) 

      return block
         :add(ShareGradInput(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, outch, stride)),'basicconcat'))
         :add(nn.CAddTable(true))
   end



   -- Creates count residual blocks with specified number of features
   local function layer(block, infeatures, outfeatures, count, stride)
      local s = nn.Sequential()
      for i=1,count do

         s:add(ShareGradInput(nn.ConcatTable()
              :add(block(i ==1 and infeatures or outfeatures ,outfeatures, i == 1 and stride or 1,i,count))
              :add(block(i ==1 and infeatures or outfeatures ,outfeatures, i == 1 and stride or 1,i,count)),'wayconcat'))
         s:add(ShareGradInput(nn.CMaxTable(),'maxtable'))

      end
      return s
   end

   local model = nn.Sequential()
   if opt.dataset == 'imagenet' then
      -- Configurations for ResNet:
      --  num. residual blocks, num features, residual block function
      local cfg = {
         [26]  = {{2, 2, 2, 2}, 1440, bottleneck},
         [50]  = {{3, 4, 6, 3}, 1440, bottleneck},
         [101] = {{3, 4, 23, 3}, 1440, bottleneck},
         [152] = {{3, 8, 36, 3}, 1440, bottleneck},
      }

      assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
      local def, nFeatures, block = table.unpack(cfg[depth])
      iChannels = 64
      print(' | ResNet-' .. depth .. ' ImageNet')

      -- The ResNet ImageNet model
      model:add(Convolution(3,64,7,7,2,2,3,3))
      model:add(SBatchNorm(64))
      model:add(ReLU(true))
      model:add(Max(3,3,2,2,1,1))
      model:add(layer(block, 45, def[1]))
      model:add(layer(block, 90, def[2], 2))
      model:add(layer(block, 180, def[3], 2))
      model:add(layer(block, 360, def[4], 2))
      model:add(Avg(7, 7, 1, 1))
      model:add(nn.View(nFeatures):setNumInputDims(3))
      model:add(nn.Linear(nFeatures, 1000))

   elseif opt.dataset == 'cifar10' then
      -- Model type specifies number of layers for CIFAR-10 model
      assert((depth - 2) % 9 == 0, 'depth should be one of 20, 32, 44, 56, 110, 1202')
      local n = (depth - 2) / 9
      --iChannels = 16*4
      print(' | ResNet-' .. depth .. ' CIFAR-10' .. ' layers:' .. n)
      local wide = 1*4
      -- The ResNet CIFAR-10 model
      model:add(Convolution(3,16,3,3,1,1,1,1))
      model:add(SBatchNorm(16))
      model:add(ReLU(true))
      

      model:add(layer(bottleneck, 16,  12*wide, n, 1))
      model:add(ShareGradInput(SBatchNorm(12*wide), 'preact'))
      model:add(ReLU(true))
      model:add(Avg(2,2,2,2))
      model:add(nn.Dropout(0.2))

      model:add(layer(bottleneck, 12*wide,  24*wide, n, 1))
      model:add(ShareGradInput(SBatchNorm(24*wide), 'preact'))  
      model:add(ReLU(true))
      model:add(Avg(2,2,2,2))       
      model:add(nn.Dropout(0.2))

      model:add(layer(bottleneck, 24*wide,  45*wide, n, 1))
      model:add(ShareGradInput(SBatchNorm(45*wide), 'last'))
      model:add(ReLU(true))
      model:add(Avg(8, 8, 1, 1))

      model:add(nn.View(45*wide):setNumInputDims(3)) 
      model:add(nn.Linear(45*wide, 10))
 
   elseif opt.dataset == 'cifar100' then
      -- Model type specifies number of layers for CIFAR-10 model
      assert((depth - 2) % 9 == 0, 'depth should be one of 20, 32, 44, 56, 110, 1202')
      local n = (depth - 2) / 9
      --iChannels = 16*4
      print(' | ResNet-' .. depth .. ' CIFAR-100' .. ' layers:' .. n)
      local wide = 1*4

      -- The ResNet CIFAR-100 model
      model:add(Convolution(3,16,3,3,1,1,1,1))
      model:add(SBatchNorm(16))
      model:add(ReLU(true))
      

      model:add(layer(bottleneck, 16,  12*wide, n, 1))
      model:add(ShareGradInput(SBatchNorm(12*wide), 'preact'))
      model:add(ReLU(true))
      model:add(Avg(2,2,2,2))
      model:add(nn.Dropout(0.2))

      concate = nn.Concat(2)
      concate:add(nn.Identity()) --transition
      a = nn.Sequential()

       a:add(layer(bottleneck, 12*wide,  24*wide, n, 1))
      concate:add(a)
      model:add(concate)
      model:add(ShareGradInput(SBatchNorm(36*wide), 'preact'))  
      model:add(ReLU(true))
      model:add(Avg(2,2,2,2))       
      model:add(nn.Dropout(0.2))

      concateF = nn.Concat(2)
      concateF:add(nn.Identity()) --transition
      b = nn.Sequential()
       b:add(layer(bottleneck, 36*wide,  45*wide, n, 1))
      concateF:add(b)
      model:add(concateF)
      model:add(ShareGradInput(SBatchNorm(81*wide), 'last'))
      model:add(ReLU(true))
      model:add(Avg(8, 8, 1, 1))

      model:add(nn.View(81*wide):setNumInputDims(3)) 
      model:add(nn.Linear(81*wide, 100))
      --print(model)
      --print(model:getParameters():size(1))

   elseif opt.dataset == 'svhn' then
      -- Model type specifies number of layers for CIFAR-10 model
      assert((depth - 2) % 9 == 0, 'depth should be one of 20, 32, 44, 56, 110, 1202')
      local n = (depth - 2) / 9
      --iChannels = 16*4
      print(' | ResNet-' .. depth .. ' svhn' .. ' layers:' .. n)
      local wide = 2*4

      model:add(Convolution(3,16,3,3,1,1,1,1))
      model:add(SBatchNorm(16))
      model:add(ReLU(true))
      

      model:add(layer(bottleneck, 16,  12*wide, n, 1))
      model:add(ShareGradInput(SBatchNorm(12*wide), 'preact'))
      model:add(ReLU(true))
      model:add(Avg(2,2,2,2))
      model:add(nn.Dropout(0.2))  

      concate = nn.Concat(2)
      concate:add(nn.Identity()) --transition

      b = nn.Sequential() 
      b:add(layer(bottleneck, 12*wide,  24*wide, n, 1))
      concate:add(b)

      model:add(concate)
      model:add(ShareGradInput(SBatchNorm(36*wide), 'preact'))  
      model:add(ReLU(true))
      model:add(Avg(2,2,2,2))       
      model:add(nn.Dropout(0.2)) 

      concateF = nn.Concat(2)
      concateF:add(nn.Identity()) --transition
 
      c = nn.Sequential()
      c:add(layer(bottleneck, 36*wide,  45*wide, n, 1))

      concateF:add(c)

      model:add(concateF)
      model:add(ShareGradInput(SBatchNorm(81*wide), 'last'))
      model:add(ReLU(true))
      model:add(Avg(8, 8, 1, 1))

      model:add(nn.View(81*wide):setNumInputDims(3)) 
      model:add(nn.Linear(81*wide, 10))
      --print(model)
      
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
