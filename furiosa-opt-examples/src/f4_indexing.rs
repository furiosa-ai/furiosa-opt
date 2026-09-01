use furiosa_opt_std::prelude::*;

axes![N = 64];

type Chip = m![1];

#[device(chip = 1)]
pub fn f4_indexing(ctx: &mut Context, input: &HbmTensor<f4e2m1, Chip, m![N]>) -> HbmTensor<f4e2m1, Chip, m![N]> {
    let mut output = HbmTensor::<f4e2m1, Chip, m![N]>::new();
    for i in 0..2 {
        let offset = i * 32;
        let input = input.view().tile::<m![N], 32, m![N = 32 # 64]>(offset);
        let output_view = output.view_mut().tile::<m![N], 32, m![N = 32 #{!} 64]>((1 - i) * 32);
        input.to_hbm_view(&mut ctx.tdma, output_view);
    }
    output
}
