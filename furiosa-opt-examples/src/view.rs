pub mod simpl {
    use furiosa_opt_std::prelude::*;

    axes![A = 512, B = 8];

    #[device(chip = 1)]
    pub fn view_simpl(
        ctx: &mut Context,
        input_hbm: &HbmTensor<i32, m![1], m![A, B]>,
    ) -> HbmTensor<i32, m![1], m![A, B]> {
        // Transfer input tensor to DM with shape:
        // - (A=512/256) on the cluster dimension,
        // - (A=512%256) on the slice dimension,
        // - (B=8) on the element dimension.
        let input = input_hbm.to_dm::<m![A / 256], m![A % 256], m![B]>(&mut ctx.tdma);

        // Allocate output tensor in DM with the same shape as input.
        let mut output = DmTensor::<i32, m![1], m![A / 256], m![A % 256], m![B]>::new();

        // Create views on input (B=0,1,2,3) and output (B=2,3,4,5), and copy data from input to output.
        let input0123 = input.view().tile::<m![B], 4, m![B = 4 # 8]>(0);
        let output2345 = output.view_mut().tile::<m![B], 4, m![B = 4 #{!} 8]>(2);
        input0123.to_dm_view(&mut ctx.tdma, output2345);

        // Create views on input (B=4,5) and output (B=6,7), and copy data from input to output.
        let input45 = input.view().tile::<m![B], 2, m![B = 2 # 8]>(4);
        let output67 = output.view_mut().tile::<m![B], 2, m![B = 2 #{!} 8]>(6);
        input45.to_dm_view(&mut ctx.tdma, output67);

        // Create views on input (B=6,7) and output (B=0,1), and copy data from input to output.
        let input67 = input.view().tile::<m![B], 2, m![B = 2 # 8]>(6);
        let output01 = output.view_mut().tile::<m![B], 2, m![B = 2 #{!} 8]>(0);
        input67.to_dm_view(&mut ctx.tdma, output01);

        // Transfer output tensor back to HBM.
        output.to_hbm(&mut ctx.tdma)
    }
}

pub mod nested {
    use furiosa_opt_std::prelude::*;

    axes![A = 512, B = 528];

    #[device(chip = 1)]
    pub fn view_nested(
        ctx: &mut Context,
        input_hbm: &HbmTensor<i32, m![1], m![A, B]>,
    ) -> HbmTensor<i32, m![1], m![A, B]> {
        let input = input_hbm.to_dm::<m![A / 256], m![A % 256], m![B]>(&mut ctx.tdma);
        let mut output = DmTensor::<i32, m![1], m![A / 256], m![A % 256], m![B]>::new();

        // Define the whole output first (full copy) so no position is left `Uninit`.
        input.view().to_dm_view(&mut ctx.tdma, output.view_mut());

        let in_nested = input
            .view()
            .tile::<m![B], 264, m![B = 264 # 528]>(0)
            .tile::<m![B = 264 # 528], 132, m![B = 132 # 528]>(0);
        let out_nested = output
            .view_mut()
            .tile::<m![B], 264, m![B = 264 #{!} 528]>(0)
            .tile::<m![B = 264 #{!} 528], 132, m![B = 132 #{!} 528]>(0);
        in_nested.to_dm_view(&mut ctx.tdma, out_nested);

        output.to_hbm(&mut ctx.tdma)
    }
}

pub mod padding {
    use furiosa_opt_std::prelude::*;

    axes![A = 9, B = 7];
}

pub mod whole_view_mut {
    use furiosa_opt_std::prelude::*;

    axes![S = 128];

    #[device(chip = 1)]
    pub fn write_view_mut(
        ctx: &mut Context,
        input: &HbmTensor<bf16, m![1], m![S]>,
        output: HbmTensorViewMut<'_, bf16, m![1], m![S]>,
    ) {
        let dm: DmTensor<bf16, m![1], m![1 # 2], m![1 # 256], m![S]> = input.to_dm(&mut ctx.tdma);
        dm.view().to_hbm_view(&mut ctx.tdma, output);
    }
}
