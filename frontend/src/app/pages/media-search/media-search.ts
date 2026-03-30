import { Component, OnDestroy, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subscription, interval } from 'rxjs';
import { switchMap, takeWhile } from 'rxjs/operators';

export type MediaType = 'anime' | 'tv' | 'movie' | 'game' | 'book';

export interface MediaJob {
  id:                 number;
  title:              string;
  media_type:         MediaType;
  media_type_display: string;
  wiki_slug:          string;
  wiki_url:           string;
  status:             'p' | 'pr' | 'c' | 'f';
  status_display:     string;
  error_message:      string;
  submitted_at:       string;
  completed_at:       string | null;
}

const MEDIA_OPTIONS: { value: MediaType; label: string; emoji: string }[] = [
  { value: 'anime', label: 'Anime',   emoji: '⛩️' },
  { value: 'tv',    label: 'TV Show', emoji: '📺' },
  { value: 'movie', label: 'Movie',   emoji: '🎬' },
  { value: 'game',  label: 'Game',    emoji: '🎮' },
  { value: 'book',  label: 'Book',    emoji: '📚' },
];

const EXAMPLES: { title: string; type: MediaType }[] = [
  { title: 'Breaking Bad',    type: 'tv'    },
  { title: 'Attack on Titan', type: 'anime' },
  { title: 'The Witcher',     type: 'game'  },
  { title: 'Dune',            type: 'book'  },
];

@Component({
  selector:    'app-media-search',
  standalone:  true,
  imports:     [CommonModule, FormsModule],
  templateUrl: './media-search.html',
  styleUrl:    './media-search.css',
})
export class MediaSearch implements OnDestroy {
  readonly mediaOptions = MEDIA_OPTIONS;
  readonly examples     = EXAMPLES;

  // Form state
  title     = signal('');
  mediaType = signal<MediaType>('anime');

  // Job state
  job        = signal<MediaJob | null>(null);
  pastJobs   = signal<MediaJob[]>([]);
  submitting = signal(false);
  error      = signal<string | null>(null);

  private readonly API  = '/api/media/';
  private pollSub?: Subscription;

  constructor(private http: HttpClient) {}

  // ── Form ────────────────────────────────────────────────────────────────

  setType(t: MediaType) { this.mediaType.set(t); }

  tryExample(ex: { title: string; type: MediaType }) {
    this.title.set(ex.title);
    this.mediaType.set(ex.type);
  }

  onKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter') this.submit();
  }

  submit() {
    const t = this.title().trim();
    if (!t || this.submitting()) return;

    this.submitting.set(true);
    this.error.set(null);
    this.job.set(null);
    this.stopPolling();

    this.http.post<MediaJob>(`${this.API}submit/`, {
      title:      t,
      media_type: this.mediaType(),
    }).subscribe({
      next: job => {
        this.job.set(job);
        this.submitting.set(false);
        this.startPolling(job.id);
      },
      error: err => {
        this.error.set(err.error?.error ?? 'Submission failed. Please try again.');
        this.submitting.set(false);
      },
    });
  }

  // ── Polling ──────────────────────────────────────────────────────────────

  private startPolling(jobId: number) {
    this.pollSub = interval(3000).pipe(
      switchMap(() => this.http.get<MediaJob>(`${this.API}${jobId}/`)),
      takeWhile(job => job.status === 'p' || job.status === 'pr', /* inclusive */ true),
    ).subscribe({
      next: job => {
        this.job.set(job);
        if (job.status === 'c' || job.status === 'f') {
          this.pastJobs.update(list => [job, ...list.filter(j => j.id !== job.id)]);
          this.stopPolling();
        }
      },
    });
  }

  private stopPolling() {
    this.pollSub?.unsubscribe();
    this.pollSub = undefined;
  }

  // ── Helpers ──────────────────────────────────────────────────────────────

  reset() {
    this.stopPolling();
    this.job.set(null);
    this.error.set(null);
    this.title.set('');
  }

  graphUrl(job: MediaJob): string {
    return `/graph/${job.id}`;
  }

  statusClass(s: MediaJob['status']): string {
    return `badge badge--${s}`;
  }

  isTerminal(s: MediaJob['status']): boolean {
    return s === 'c' || s === 'f';
  }

  ngOnDestroy() { this.stopPolling(); }
}
