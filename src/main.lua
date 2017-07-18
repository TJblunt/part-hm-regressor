require 'paths'
paths.dofile('ref.lua')         -- Parse command line input and do global variable initialization
paths.dofile('model.lua')       -- Read in network model(train_co.lua delete)
paths.dofile('train_reg.lua')   -- Load up training/testing functions
                                -- modify to train part detector(train_det.lua)/heatmap regressor(train_reg.lua)/part heatmap regressor(train.lua)
                                -- note:select drawGaussian/drawbinarymap in generateSample()

-- Set up data loader
torch.setnumthreads(1)
local Dataloader = paths.dofile('util/dataloader.lua')
loader = Dataloader.create(opt, dataset, ref)

-- Initialize logs
ref.log = {}
ref.log.train = Logger(paths.concat(opt.save, 'train.log'), opt.continue)
ref.log.valid = Logger(paths.concat(opt.save, 'valid.log'), opt.continue)

-- Main training loop
for i=1,opt.nEpochs do
    print("==> Starting epoch: " .. epoch .. "/" .. (opt.nEpochs + opt.epochNumber - 1))
    if opt.trainIters > 0 then train() end
    if opt.validIters > 0 then valid() end
    epoch = epoch + 1
    collectgarbage()
end

-- Update reference for last epoch
opt.lastEpoch = epoch - 1

-- Save model
model:clearState()
torch.save(paths.concat(opt.save,'options.t7'), opt)
torch.save(paths.concat(opt.save,'optimState.t7'), optimState)
torch.save(paths.concat(opt.save,'final_model.t7'), model)

-- Generate final predictions on validation set
if opt.finalPredictions then
	ref.log = {}
	loader.test = Dataloader(opt, dataset, ref, 'test')
	predict()
end
