css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex;
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.bot {
    background-color: #475063;
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 100px;  /* Increased from 78px to 100px */
  max-height: 100px; /* Increased from 78px to 100px */
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cpng.pikpng.com/pngl/s/398-3982669_llama-llama-profile-pic-llama-profile-clipart.png" style="max-height: 100px; max-width: 100px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://freesvg.org/img/abstract-user-flat-4.png" style="max-height: 100px; max-width: 100px; border-radius: 50%; object-fit: cover;">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''