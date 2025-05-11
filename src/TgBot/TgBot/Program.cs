using System.Net.Http.Headers;
using System.Text.Json;
using Telegram.Bot;
using Telegram.Bot.Polling;
using Telegram.Bot.Types;
using Telegram.Bot.Types.Enums;


string json = await File.ReadAllTextAsync("appsettings.json");
using var doc = JsonDocument.Parse(json);
string token = doc.RootElement.GetProperty("BotToken").GetString();


using var cts = new CancellationTokenSource();
var bot = new TelegramBotClient(token, cancellationToken: cts.Token);
var me = await bot.GetMe();
bot.OnError += OnError;
bot.OnMessage += OnMessage;
bot.OnUpdate += OnUpdate;

while (true) { }


async Task OnMessage(Message msg, UpdateType type)
{
    if (msg.Text == "/start")
    {
        await bot.SendMessage(msg.Chat, "Отпрвьте видео для проверки на наличие deepfake");
    }
    else if (msg.Video != null)
    { 
        await using var fileStream = new MemoryStream();
        var tgFile = await bot.GetInfoAndDownloadFile(msg.Video.FileId, fileStream);
        var result = await SendVideoToBackend(fileStream);
        await bot.SendMessage(msg.Chat, result);
    }
    else
    {
        await bot.SendMessage(msg.Chat, "Вы должны отправить видео");
    }
}
async Task<string> SendVideoToBackend(MemoryStream videoStream)
{
    using (var client = new HttpClient())
    {
        var form = new MultipartFormDataContent();

        var fileContent = new StreamContent(videoStream);
        fileContent.Headers.ContentType = new MediaTypeHeaderValue("video/mp4"); 

        form.Add(fileContent, "file", "video.mp4");

        var response = await client.PostAsync("http://backend:8080/upload", form);
        var responseBody = await response.Content.ReadAsStringAsync();
        return responseBody; 
       
    }
}
async Task OnError(Exception exception, HandleErrorSource source)
{
    Console.WriteLine(exception);
}
async Task OnUpdate(Update update)
{
   
}