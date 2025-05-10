using System.Diagnostics;
using System.Net.Http.Headers;
using Microsoft.AspNetCore.Http;
using Services.Interfaces;

namespace Services
{
    public class RunModelService : IRunModelService
    {
        public async Task<int> Run(string filePath)
        {
            var process = new Process();
            process.StartInfo.FileName = "/app/venv/bin/python";
            process.StartInfo.Arguments = $"/app/Model/src/main.py \"{filePath}\"";
            process.StartInfo.UseShellExecute = false;

            process.Start();
            await process.WaitForExitAsync();

            return process.ExitCode;
        }
    }
}
