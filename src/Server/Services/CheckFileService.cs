using Microsoft.AspNetCore.Http;
using Services.Interfaces;

namespace Services
{
    public class CheckFileService : ICheckFileService
    {
        public async Task<bool> IsVideoFileAsync(IFormFile file)
        {
            using var stream = file.OpenReadStream();
            byte[] header = new byte[20];
            int bytesRead = await stream.ReadAsync(header, 0, header.Length);

            if (bytesRead < 12) return false;

            string hex = BitConverter.ToString(header).Replace("-", "").ToUpper();

            // MP4, MOV, 3GP → ftyp
            if (hex.Contains("66747970")) return true;

            // AVI → RIFF....AVI
            if (hex.StartsWith("52494646") && hex.Substring(16, 8) == "41564920") return true;

            // MKV / WebM → 1A45DFA3
            if (hex.StartsWith("1A45DFA3")) return true;

            // FLV → 464C56
            if (hex.StartsWith("464C56")) return true;

            // MPEG → 000001BA or 000001B3
            if (hex.StartsWith("000001BA") || hex.StartsWith("000001B3")) return true;

            // WMV / ASF → 3026B2758E66CF11A6D9
            if (hex.StartsWith("3026B2758E66CF11A6D9")) return true;

            return false;
        }
    }
}
