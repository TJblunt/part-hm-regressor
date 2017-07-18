require 'paths'
require 'sys'
projectDir = paths.concat('/home/jtang/pose_estimation/','pose-hg-train')
paths.dofile(projectDir .. '/src/ref.lua')

   -- Set up input image
    local im = image.load('images/' .. a['images'][idxs[i]])
    local center = a['center'][idxs[i]]
    local scale = a['scale'][idxs[i]]
    local inp = crop(im, center, scale, 0, 256)

    -- Get network output
    local out = m:forward(inp:view(1,3,256,256):cuda())
    local hm = out[#out][1]:float()
    hm[hm:lt(0)] = 0

    -- Get predictions (hm and img refer to the coordinate space)
    local preds_hm, preds_img = getPreds(hm, center, scale)
    preds[i]:copy(preds_img)

    xlua.progress(i,nsamples)

    -- Display the result
    if arg[1] == 'demo' then
        preds_hm:mul(4) -- Change to input scale
        local dispImg = drawOutput(inp, hm, preds_hm[1])
        w = image.display{image=dispImg,win=w}
        sys.sleep(3)
    end

