import { Component, OnInit, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';

interface EpubFile {
  id: number;
  original_filename: string;
  uploaded_at: string;
  status: 'p' | 'pr' | 'c' | 'f';
}

const STATUS_LABELS: Record<string, string> = {
  p: 'pending',
  pr: 'processing',
  c: 'done',
  f: 'failed',
};

@Component({
  selector: 'app-epub-list',
  imports: [CommonModule],
  templateUrl: './epub-list.html',
  styleUrl: './epub-list.css'
})
export class EpubList implements OnInit {
  epubs   = signal<EpubFile[]>([]);
  loading = signal(true);
  error   = signal<string | null>(null);

  private readonly API = '/api/epubs/';

  constructor(private http: HttpClient) {}

  ngOnInit() {
    this.http.get<EpubFile[]>(this.API).subscribe({
      next:  data => { this.epubs.set(data); this.loading.set(false); },
      error: ()   => { this.error.set('Could not reach the API.'); this.loading.set(false); },
    });
  }

  stripExtension(name: string): string {
    return name.replace(/\.[^.]+$/, '');
  }

  statusLabel(s: string): string {
    return STATUS_LABELS[s] ?? s;
  }
}
