-- Prepare tensors for saving network output
local validSamples = opt.validIters * opt.validBatch
saved = {idxs = torch.Tensor(validSamples),
         preds = torch.Tensor(validSamples, unpack(ref.predDim))}
if opt.saveInput then saved.input = torch.Tensor(validSamples, unpack(ref.inputDim)) end
if opt.saveHeatmaps then saved.heatmaps = torch.Tensor(validSamples, unpack(ref.outputDim[1])) end

-- Main processing step
function step(tag)
    local avgLoss, avgAcc = 0.0, 0.0
    local output, err, idx
    local param, gradparam = model:getParameters()
    local function evalFn(x) return criterion.output, gradparam end
    model_det=torch.load('/home/jtang/pose_estimation/pose-hg-train/exp/mpii/test-n1/model_20.t7')

    if tag == 'train' then
        model:training()
        set = 'train'
    else
        model:evaluate()
        if tag == 'predict' then
            print("==> Generating predictions...")
            local nSamples = dataset:size('test')
            saved = {idxs = torch.Tensor(nSamples),
                     preds = torch.Tensor(nSamples, unpack(ref.predDim))}
            if opt.saveInput then saved.input = torch.Tensor(nSamples, unpack(ref.inputDim)) end
            if opt.saveHeatmaps then saved.heatmaps = torch.Tensor(nSamples, unpack(ref.outputDim[1])) end
            set = 'test'
        else
            set = 'valid'
        end
    end

    local nIters = opt[set .. 'Iters']
    local bs = opt[set .. 'Batch']
    for i,sample in loader[set]:run() do
        xlua.progress(i, nIters)
        local input, label, indices = unpack(sample)
        --print(input:size())
        --print(label:size())

        if opt.GPU ~= -1 then
            -- Convert to CUDA
            input = applyFn(function (x) return x:cuda() end, input)
            label = applyFn(function (x) return x:cuda() end, label)
        end
        -- implement part heatmap detector
        
        local output_det= model_det:forward(input)
        local input_det=torch.Tensor(bs,19,256,256):zero()
        local hm_det=torch.Tensor(bs,16,64,64):zero()
        if type(output_det)=='table' then
            hm_det=output_det[#output_det]:float()
        else
            hm_det=output_det:float()
        end
        input=input:float()
        for i=1,bs do
            for j=1,16 do
                input_det[i][j]=image.scale(hm_det[i][j],256)
            end
            input_det[i][17]=input[i][1]
            input_det[i][18]=input[i][2]
            input_det[i][19]=input[i][3]
        end
        input_det=applyFn(function (x) return x:cuda() end, input_det)

        -- Do a forward pass and calculate loss
        local output = model:forward(input_det)
        local err = criterion:forward(output, label)
        avgLoss = avgLoss + err / nIters

        if tag == 'train' then
            -- Training: Do backpropagation and optimization
            model:zeroGradParameters()
            model:backward(input_det, criterion:backward(output, label))
            optfn(evalFn, param, optimState)
        else
            -- Validation: Get flipped output
            output = applyFn(function (x) return x:clone() end, output)
            input = applyFn(function (x) return x:cuda() end, input)
            local flippedinput=flip(input)
            local flippedoutput_det = model_det:forward(flippedinput)
            local flippedinput_det=torch.Tensor(bs,19,256,256):zero()
            local flippedhm_det=torch.Tensor(bs,16,64,64):zero()
            if type(flippedoutput_det)=='table' then
                flippedhm_det=flippedoutput_det[#flippedoutput_det]:float()
            else
                flippedhm_det=flippedoutput_det:float()
            end
            flippedinput=flippedinput:float()
            for i=1,bs do
                for j=1,16 do
                    flippedinput_det[i][j]=image.scale(flippedhm_det[i][j],256)
                end
                flippedinput_det[i][17]=flippedinput[i][1]
                flippedinput_det[i][18]=flippedinput[i][2]
                flippedinput_det[i][19]=flippedinput[i][3]
            end
            flippedinput_det=applyFn(function (x) return x:cuda() end, flippedinput_det)

            -- Do a forward pass and calculate loss
            local flippedOut = model:forward(flippedinput_det)
            flippedOut = applyFn(function (x) return flip(shuffleLR(x)) end, flippedOut)
            output = applyFn(function (x,y) return x:add(y):div(2) end, output, flippedOut)
            

            -- Save sample
            -- local bs = opt[set .. 'Batch']
            local tmpIdx = (i-1) * bs + 1
            local tmpOut = output
            if type(tmpOut) == 'table' then tmpOut = output[#output] end
            if opt.saveInput then saved.input:sub(tmpIdx, tmpIdx+bs-1):copy(input) end
            if opt.saveHeatmaps then saved.heatmaps:sub(tmpIdx, tmpIdx+bs-1):copy(tmpOut) end
            saved.idxs:sub(tmpIdx, tmpIdx+bs-1):copy(indices)
            saved.preds:sub(tmpIdx, tmpIdx+bs-1):copy(postprocess(set,indices,output))
        end

        -- Calculate accuracy
        avgAcc = avgAcc + accuracy(output, label) / nIters
    end


    -- Print and log some useful metrics
    print(string.format("      %s : Loss: %.7f Acc: %.4f"  % {set, avgLoss, avgAcc}))
    if ref.log[set] then
        table.insert(opt.acc[set], avgAcc)
        ref.log[set]:add{
            ['epoch     '] = string.format("%d" % epoch),
            ['loss      '] = string.format("%.6f" % avgLoss),
            ['acc       '] = string.format("%.4f" % avgAcc),
            ['LR        '] = string.format("%g" % optimState.learningRate)
        }
    end

    if (tag == 'valid' and opt.snapshot ~= 0 and epoch % opt.snapshot == 0) or tag == 'predict' then
        -- Take a snapshot
        model:clearState()
        torch.save(paths.concat(opt.save, 'options.t7'), opt)
        torch.save(paths.concat(opt.save, 'optimState.t7'), optimState)
        torch.save(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model)
        local predFilename = 'preds.h5'
        if tag == 'predict' then predFilename = 'final_' .. predFilename end
        local predFile = hdf5.open(paths.concat(opt.save,predFilename),'w')
        for k,v in pairs(saved) do predFile:write(k,v) end
        predFile:close()
    end
end

function train() step('train') end
function valid() step('valid') end
function predict() step('predict') end
