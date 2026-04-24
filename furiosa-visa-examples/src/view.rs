pub mod simpl {
    use furiosa_visa_std::prelude::*;

    axes![A = 512, B = 4];

    #[device(chip = 1)]
    pub fn view_simpl(
        ctx: &mut Context,
        input_hbm: &HbmTensor<i32, m![1], m![A, B]>,
    ) -> HbmTensor<i32, m![1], m![A, B]> {
        // Transfer input tensor to DM with shape:
        // - (A=512/256) on the cluster dimension,
        // - (A=512%256) on the slice dimension,
        // - (B=4) on the element dimension.
        let input = input_hbm.to_dm::<m![A / 256], m![A % 256], m![B]>(&mut ctx.tdma, 0x2000);

        // Allocate output tensor in DM with the same shape as input.
        let mut output = unsafe { DmTensor::<i32, m![1], m![A / 256], m![A % 256], m![B]>::from_addr(0x3000) };

        // Create views on input (B=0,1,2) and output (B=1,2,3), and copy data from input to output.
        let input012 = input.view().tile::<m![B], 3, m![B = 3 # 4]>(0);
        let output123 = output.view_mut().tile::<m![B], 3, m![B = 3 # 4]>(1);
        input012.to_dm_view(&mut ctx.tdma, output123);

        // Create views on input (B=3) and output (B=0), and copy data from input to output.
        let input3 = input.view().tile::<m![B], 1, m![1 # 4]>(3);
        let output0 = output.view_mut().tile::<m![B], 1, m![1 # 4]>(0);
        input3.to_dm_view(&mut ctx.tdma, output0);

        // Transfer output tensor back to HBM.
        output.to_hbm(&mut ctx.tdma, 0x4000)
    }
}

pub mod padding {
    use furiosa_visa_std::prelude::*;

    axes![A = 9, B = 7];

    #[device(chip = 1)]
    pub fn view_padding(
        ctx: &mut Context,
        input_hbm: &HbmTensor<i32, m![1], m![[A, B] # 64]>,
    ) -> HbmTensor<i32, m![1], m![[A, B] # 64]> {
        // Transfer input tensor to DM with shape:
        // - 1 on the cluster dimension,
        // - (A=9)(B=7)=64/16 on the slice dimension,
        // - ((A=9)(B=7)=64%8) * ((A=9)(B=7)=64/8%2) on the element dimension.
        let input = input_hbm
            .to_dm::<m![1], m![[A, B] # 64 / 16], m![[A, B] # 64 % 8, [A, B] # 64 / 8 % 2]>(&mut ctx.tdma, 0x2000);

        // Allocate output tensor in DM with the same shape as input.
        let mut output = unsafe {
            DmTensor::<i32, m![1], m![1], m![[A, B] # 64 / 16], m![[A, B] # 64 % 8, [A, B] # 64 / 8 % 2]>::from_addr(
                0x3000,
            )
        };

        // Create views on input (((A=9)(B=7)=64%4)=0,1,2) and output (((A=9)(B=7)=64%4)=1,2,3), and copy data from input to output.
        let input012 = input
            .view()
            .tile::<m![[A, B] # 64 % 4], 3, m![[A, B] # 64 / 4 % 2, ([A, B] # 64 % 4) = 3 # 4, [A, B] # 64 / 8 % 2]>(0);
        let output123 = output
            .view_mut()
            .tile::<m![[A, B] # 64 % 4], 3, m![[A, B] # 64 / 4 % 2, ([A, B] # 64 % 4) = 3 # 4, [A, B] # 64 / 8 % 2]>(1);
        input012.to_dm_view(&mut ctx.tdma, output123);

        // Create views on input (((A=9)(B=7)=64%4)=3) and output (((A=9)(B=7)=64%4)=0), and copy data from input to output.
        let input3 = input
            .view()
            .tile::<m![[A, B] # 64 % 4], 1, m![[A, B] # 64 / 4 % 2, 1 # 4, [A, B] # 64 / 8 % 2]>(3);
        let output0 = output
            .view_mut()
            .tile::<m![[A, B] # 64 % 4], 1, m![[A, B] # 64 / 4 % 2, 1 # 4, [A, B] # 64 / 8 % 2]>(0);
        input3.to_dm_view(&mut ctx.tdma, output0);

        // Transfer output tensor back to HBM.
        output.to_hbm(&mut ctx.tdma, 0x4000)
    }
}
