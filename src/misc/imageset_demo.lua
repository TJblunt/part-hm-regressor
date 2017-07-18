require 'paths'
require 'sys'
projectDir = paths.concat('/home/jtang/pose_estimation/','pose-hg-train')
paths.dofile(projectDir .. '/src/ref.lua')

--local set = 'train'
--local tmpIdx = opt.idxRef[set][torch.random(dataset:size(set))]
--print(tmpIdx)
--hms,inp = heatmapVisualization(set,tmpIdx)
--image.display(inp)
--image.display(hms)

function loadPred(predFile, doHm, doInp)
    local f = hdf5.open(predFile,'r')
    local inp,hms
    local idxs = f:read('idxs'):all()
    local preds = f:read('preds'):all()
    if doHm then hms = f:read('heatmaps'):all() end
    if doInp then inp = f:read('input'):all() end
    return idxs, preds, hms, inp
end

idxs, preds, hms, inp = loadPreds('mpii/test-run/preds', false, false)
local tmpIdx = torch.random(idxs:size(1))
local img = dataset:loadImage(idxs[tmpIdx]) -- Load original image
print(tmpIdx)
drawSkeleton(img, preds[tmpIdx]:narrow(2,1,2):clone(), preds[tmpIdx]:narrow(2,5,1):clone():view(-1))

print("Predicted pose:"); sys.sleep(.01)
image.display(img); sys.sleep(.01)

if hms then
    -- Prepare heatmap visualization
  local hmImg = heatmapVisualization(nil,idxs[tmpIdx],hms[tmpIdx])
  print("Heatmaps: (red - ground truth, blue - predicted)"); sys.sleep(.01)
  image.display(hmImg)
end