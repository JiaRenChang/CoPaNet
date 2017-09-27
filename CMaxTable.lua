local CMaxTable, parent = torch.class('nn.CMaxTable', 'nn.Module')

function CMaxTable:__init(soft)
   parent.__init(self)
   self.gradInput = {}
   --self.gradMask  ={}
end

function CMaxTable:updateOutput(input)
   
   self.output:resizeAs(input[1]):copy(input[1])
   self.expSum = input[1]:clone():zero()
   
   for i=2,#input do
      self.output:cmax(input[i])
   end

   --for i=1,#input do
   --      self.gradMask[i] = torch.eq(self.output,input[i])
  -- end

   return self.output
end

function CMaxTable:updateGradInput(input, gradOutput)
   for i=1,#input do
         self.gradInput[i] = self.gradInput[i] or input[1].new()
         self.gradInput[i]:resizeAs(input[i]):eq(self.output,input[i])
         self.gradInput[i]:cmul(gradOutput)
   end

   for i=#input+1, #self.gradInput do
       self.gradInput[i] = nil
   end

   return self.gradInput
end
